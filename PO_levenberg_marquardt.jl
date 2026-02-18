using LinearAlgebra,StaticArrays,Random,Base.Threads,Printf,CairoMakie

# --------------------------
# helpers
# --------------------------
const Vec3=SVector{3,Float64}
const Vec6=SVector{6,Float64}
const _TWO_PI=2*pi
@inline normalize3(v::Vec3)=(n=norm(v);n==0.0 ? v : v/n)
@inline wrap_pi(x::Float64)=(y=mod(x+pi,_TWO_PI)-pi;return y)
@inline φ_of(s::Vec3)=atan(s[2],s[1])
@inline θ2_of(s2::Vec3)=atan(s2[2],s2[1])
@inline phase_g(s2::Vec3,θ2sec::Float64)=wrap_pi(θ2_of(s2)-θ2sec)
@inline cross_ex(v::Vec3)=Vec3(0.0,-v[3],v[2]) # e_x × v

"""
    H(s1::Vec3, s2::Vec3, α::Float64) -> Float64

Classical Hamiltonian (energy) for two coupled spins.
- `H = s1_z + s2_z + α * s1_x * s2_x`.

Here `α` is the coupling strength (often denoted λ elsewhere).
Used as an energy constraint in the multiple-shooting residual.
"""
@inline function H(s1::Vec3,s2::Vec3,α::Float64)
    return s1[3]+s2[3]+α*s1[1]*s2[1]
end
@inline function rot_x(v::Vec3,θ::Float64)
    s,c=sincos(θ)
    return Vec3(v[1],c*v[2]-s*v[3],s*v[2]+c*v[3])
end
@inline function rot_z(v::Vec3,θ::Float64)
    s,c=sincos(θ)
    return Vec3(c*v[1]-s*v[2],s*v[1]+c*v[2],v[3])
end

#############################
# Strang split flows
#############################
"""
    flow_A!(s1::MVector{3,Float64}, s2::MVector{3,Float64}, h::Float64) -> nothing

In-place Strang "A" subflow: rotate both spins about the z-axis by angle `h`.

Arguments
- `s1`, `s2`: mutable 3-vectors (on/near the unit sphere).
- `h`: substep size (angle).

This is the integrable part corresponding to uncoupled precession around z.
"""
@inline function flow_A!(s1::MVector{3,Float64},s2::MVector{3,Float64},h::Float64)
    s1.=rot_z(Vec3(s1),h)
    s2.=rot_z(Vec3(s2),h)
    return nothing
end
"""
    flow_B!(s1::MVector{3,Float64}, s2::MVector{3,Float64}, h::Float64, α::Float64) -> nothing

In-place Strang "B" subflow: coupled x-rotations.

- Spin 1 rotates about x by angle `h * ω1` with `ω1 = α * s2_x` (pre-B state).
- Spin 2 rotates about x by angle `h * ω2` with `ω2 = α * s1_x` (pre-B state).
"""
@inline function flow_B!(s1::MVector{3,Float64},s2::MVector{3,Float64},h::Float64,α::Float64)
    v1=Vec3(s1);v2=Vec3(s2)
    ω1=α*v2[1];ω2=α*v1[1]
    s1.=rot_x(v1,h*ω1)
    s2.=rot_x(v2,h*ω2)
    return nothing
end
"""
    strang_step!(s1::MVector{3,Float64}, s2::MVector{3,Float64}, h::Float64, α::Float64; renorm::Bool=true) -> nothing

Perform one Strang-splitting step of size `h` for the coupled-spin flow.

Scheme
- `A(h/2) ∘ B(h) ∘ A(h/2)` where A is z-rotation and B is the coupled x-rotation.

Keyword
- `renorm`: if `true`, renormalize both spins to unit length after the step.

Used both for seeding (forward integration) and for segment integration in the MS residual.
"""
@inline function strang_step!(s1::MVector{3,Float64},s2::MVector{3,Float64},h::Float64,α::Float64;renorm::Bool=true)
    flow_A!(s1,s2,0.5*h);flow_B!(s1,s2,h,α);flow_A!(s1,s2,0.5*h)
    if renorm
        s1.=normalize3(Vec3(s1));s2.=normalize3(Vec3(s2))
    end
    return nothing
end

############################
# Strang tangent map
############################
"""
    Rz_apply!(x1::Float64, x2::Float64, x3::Float64, θ::Float64) -> (Float64,Float64,Float64)

Apply a z-axis rotation by angle `θ` to the components `(x1,x2,x3)`.

Returns a tuple `(y1,y2,y3)` corresponding to `rot_z(Vec3(x1,x2,x3), θ)`,
but avoids allocating a `Vec3`.

Used for column-wise updates of the 6×6 tangent matrix.
"""
@inline function Rz_apply!(x1::Float64,x2::Float64,x3::Float64,θ::Float64)
    s,c=sincos(θ)
    return (c*x1-s*x2,s*x1+c*x2,x3)
end
"""
    Rx_apply!(x1::Float64, x2::Float64, x3::Float64, θ::Float64) -> (Float64,Float64,Float64)

Apply an x-axis rotation by angle `θ` to the components `(x1,x2,x3)`.

Returns a tuple `(y1,y2,y3)` corresponding to `rot_x(Vec3(x1,x2,x3), θ)`,
but avoids allocating a `Vec3`.

Used for column-wise tangent propagation in the Strang B-step.
"""
@inline function Rx_apply!(x1::Float64,x2::Float64,x3::Float64,θ::Float64)
    s,c=sincos(θ)
    return (x1,c*x2-s*x3,s*x2+c*x3)
end
"""
    renorm3_with_jac!(s::MVector{3,Float64}, A::Matrix{Float64}, rows0::Int) -> nothing

Renormalize a 3-vector `s` to unit length and left-multiply the corresponding
3 rows of the tangent matrix `A` by the Jacobian of the normalization map.

- Normalization map: `s ↦ s / ||s||`.
- Jacobian: `J = (I - s sᵀ) / ||s||` evaluated at the normalized `s`.
- The rows updated are `rows0:rows0+2` (e.g. `1:3` for spin1, `4:6` for spin2).
"""
@inline function renorm3_with_jac!(s::MVector{3,Float64},A::Matrix{Float64},rows0::Int)
    vx=s[1];vy=s[2];vz=s[3]
    n=sqrt(vx*vx+vy*vy+vz*vz)
    if n==0.0; return nothing; end
    invn=1.0/n
    sx=vx*invn;sy=vy*invn;sz=vz*invn
    s[1]=sx;s[2]=sy;s[3]=sz
    # J = (I - s transpose(s))/n
    j11=(1.0-sx*sx)*invn
    j12=(-sx*sy)*invn
    j13=(-sx*sz)*invn
    j21=(-sy*sx)*invn
    j22=(1.0-sy*sy)*invn
    j23=(-sy*sz)*invn
    j31=(-sz*sx)*invn
    j32=(-sz*sy)*invn
    j33=(1.0-sz*sz)*invn
    r1=rows0
    r2=rows0+1
    r3=rows0+2
    @inbounds for col in 1:size(A,2)
        a1=A[r1,col];a2=A[r2,col];a3=A[r3,col]
        A[r1,col]=j11*a1+j12*a2+j13*a3
        A[r2,col]=j21*a1+j22*a2+j23*a3
        A[r3,col]=j31*a1+j32*a2+j33*a3
    end
    return nothing
end
"""
    strang_step_tangent!(s1::MVector{3,Float64}, s2::MVector{3,Float64}, A::Matrix{Float64}, h::Float64, α::Float64) -> nothing

Propagate both the state `(s1,s2)` and the 6×6 tangent matrix `A` through one
Strang step of size `h`.

- `(s1,s2)` updated by `A(h/2) ∘ B(h) ∘ A(h/2)`.
- `A` updated to the corresponding linearization (including coupling terms).
- Both spins are renormalized with consistent Jacobian updates via
  `renorm3_with_jac!`.
"""
@inline function strang_step_tangent!(s1::MVector{3,Float64},s2::MVector{3,Float64},A::Matrix{Float64},h::Float64,α::Float64)
    # Propagate (s1,s2) and tangent matrix A (6x6) through one Strang step size h.
    θz=0.5*h
    # A half: z-rotation
    v1=rot_z(Vec3(s1),θz);v2=rot_z(Vec3(s2),θz)
    s1.=v1;s2.=v2
    # A half: z-rotation (tangent), left-multiply by diag(Rz,Rz) columnwise
    @inbounds for col in 1:6
        a1=A[1,col]
        a2=A[2,col]
        a3=A[3,col]
        b1=A[4,col]
        b2=A[5,col]
        b3=A[6,col]
        (A[1,col],A[2,col],A[3,col])=Rz_apply!(a1,a2,a3,θz)
        (A[4,col],A[5,col],A[6,col])=Rz_apply!(b1,b2,b3,θz)
    end
    # B: x-rotations with coupled angles from pre-B state 
    s1x=s1[1];s2x=s2[1]
    θ1=h*α*s2x
    θ2=h*α*s1x
    k=h*α
    s1_pre=Vec3(s1)
    s2_pre=Vec3(s2)
    w1=rot_x(s1_pre,θ1)
    w2=rot_x(s2_pre,θ2)
    s1.=w1;s2.=w2
    # c1 = e_x×s1', c2 = e_x×s2' (after B)
    c1=cross_ex(Vec3(s1))
    c2=cross_ex(Vec3(s2))
    # B: tangent update columnwise 
    @inbounds for col in 1:6
        # current column = [δs1;δs2] at pre-B
        d11=A[1,col]
        d12=A[2,col]
        d13=A[3,col]
        d21=A[4,col]
        d22=A[5,col]
        d23=A[6,col]
        δs1x=d11
        δs2x=d21
        # δs1' = Rx(θ1)δs1 + k*c1*δs2x
        (u11,u12,u13)=Rx_apply!(d11,d12,d13,θ1)
        u11+=k*c1[1]*δs2x
        u12+=k*c1[2]*δs2x
        u13+=k*c1[3]*δs2x
        # δs2' = Rx(θ2)δs2 + k*c2*δs1x
        (u21,u22,u23)=Rx_apply!(d21,d22,d23,θ2)
        u21+=k*c2[1]*δs1x
        u22+=k*c2[2]*δs1x
        u23+=k*c2[3]*δs1x
        A[1,col]=u11;A[2,col]=u12;A[3,col]=u13
        A[4,col]=u21;A[5,col]=u22;A[6,col]=u23
    end
    # A half again: z-rotation (state) 
    v1=rot_z(Vec3(s1),θz)
    v2=rot_z(Vec3(s2),θz)
    s1.=v1;s2.=v2

    # A half again: z-rotation (tangent)
    @inbounds for col in 1:6
        a1=A[1,col];a2=A[2,col];a3=A[3,col]
        b1=A[4,col];b2=A[5,col];b3=A[6,col]
        (A[1,col],A[2,col],A[3,col])=Rz_apply!(a1,a2,a3,θz)
        (A[4,col],A[5,col],A[6,col])=Rz_apply!(b1,b2,b3,θz)
    end
    # keep state on sphere - sanity check
    renorm3_with_jac!(s1,A,1)
    renorm3_with_jac!(s2,A,4)
    return nothing
end
"""
    build_As_strang!(As::Vector{Matrix{Float64}}, y::Vector{Float64}, α::Float64; M::Int, nsub::Int) -> nothing

Build per-segment tangent maps `As[i]` (each 6×6) for the Strang flow.

- The multiple-shooting unknown `y` stores `M` nodes `(s1_i, s2_i)` plus the
  period `T = y[6M+1]`.
- For each segment `i`, `As[i]` approximates the derivative of the segment map
  Φ_τ with respect to the segment's initial 6 coordinates:
    `As[i] ≈ DΦ_τ( node_i )`, where `τ = T/M`.

- For each node `i` (threaded):
  1. Initialize `A = I(6)`.
  2. Integrate `nsub` Strang substeps of size `h = (τ/nsub)`,
     using `strang_step_tangent!` to advance both state and `A`.
"""
function build_As_strang!(As::Vector{Matrix{Float64}},y::Vector{Float64},α::Float64;M::Int,nsub::Int)
    T=y[6*M+1]
    τ=T/M
    s1buf=[MVector{3,Float64}(0,0,0) for _ in 1:Threads.nthreads()]
    s2buf=[MVector{3,Float64}(0,0,0) for _ in 1:Threads.nthreads()]
    Threads.@threads :static for i in 1:M
        tid=Threads.threadid()
        s1=s1buf[tid]
        s2=s2buf[tid]
        b=6*(i-1)
        s1[1]=y[b+1];s1[2]=y[b+2];s1[3]=y[b+3]
        s2[1]=y[b+4];s2[2]=y[b+5];s2[3]=y[b+6]
        A=As[i]
        fill!(A,0.0)
        @inbounds for d in 1:6
            A[d,d]=1.0
        end
        h=τ/nsub
        @inbounds for _ in 1:nsub
            strang_step_tangent!(s1,s2,A,h,α)
        end
    end
    return nothing
end
"""
    unpack(y::Vector{Float64}, M::Int) -> (s1s::Vector{Vec3}, s2s::Vector{Vec3}, T::Float64)

Unpack the multiple-shooting vector `y` into node arrays and period.

- For node `i = 1..M` (0-based block index `b = 6*(i-1)`):
  - `s1_i = (y[b+1], y[b+2], y[b+3])`
  - `s2_i = (y[b+4], y[b+5], y[b+6])`
- `T = y[6M+1]`.

Returns
- `s1s`, `s2s`: length-`M` vectors of `Vec3`.
- `T`: period.
"""
@inline function unpack(y::Vector{Float64},M::Int)
    s1s=Vector{Vec3}(undef,M); s2s=Vector{Vec3}(undef,M)
    @inbounds for i in 1:M
        b=6*(i-1)
        s1s[i]=Vec3(y[b+1],y[b+2],y[b+3])
        s2s[i]=Vec3(y[b+4],y[b+5],y[b+6])
    end
    return s1s,s2s,y[6*M+1]
end
"""
    pack!(y::Vector{Float64}, s1s::Vector{Vec3}, s2s::Vector{Vec3}, T::Float64) -> nothing

Pack node arrays and period into the multiple-shooting vector `y` in-place.

Inputs
- `s1s`, `s2s`: length-`M` vectors of node spins.
- `T`: period.

- Fills `y[1:6M]` with node coordinates and `y[6M+1]=T`.
"""
@inline function pack!(y::Vector{Float64},s1s::Vector{Vec3},s2s::Vector{Vec3},T::Float64)
    M=length(s1s)
    @inbounds for i in 1:M
        b=6*(i-1);s1=s1s[i];s2=s2s[i]
        y[b+1]=s1[1];y[b+2]=s1[2];y[b+3]=s1[3]
        y[b+4]=s2[1];y[b+5]=s2[2];y[b+6]=s2[3]
    end
    y[6*M+1]=T
    return nothing
end
"""
    renorm_spin_at!(y::Vector{Float64}, idx0::Int) -> nothing

Renormalize a single 3-component spin stored inside the packed vector `y`.

Arguments
- `idx0`: 1-based index of the x-component inside `y` (so the spin is
  `y[idx0:idx0+2]`).

- used during line search in `ms_lm_mfree_strang` to keep trial updates on the sphere.
"""
@inline function renorm_spin_at!(y::Vector{Float64},idx0::Int)
    v=normalize3(Vec3(y[idx0],y[idx0+1],y[idx0+2]))
    y[idx0]=v[1];y[idx0+1]=v[2];y[idx0+2]=v[3]
    return nothing
end
"""
    nodes_in_domain(y::Vector{Float64}, M::Int) -> Bool

Basic domain check for a packed MS vector `y`.

Returns `false` if any node spin deviates too far from unit length, otherwise `true`. This should not happen if the solver is working correctly, but it can catch grossly infeasible iterates during line search.

Note
- This is a lightweight guard used during line search; it does not enforce
  exact normalization (that is done by `renorm_spin_at!` and `strang_step!`).
"""
@inline function nodes_in_domain(y::Vector{Float64},M::Int)
    @inbounds for i in 1:M
        b=6*(i-1)
        s1=Vec3(y[b+1],y[b+2],y[b+3])
        s2=Vec3(y[b+4],y[b+5],y[b+6])
        (abs(norm(s1)-1.0)>1.0 || abs(norm(s2)-1.0)>1.0) && return false
    end
    return true
end

############################
# Segment map Φ_τ (Strang only)
############################
"""
    integrate_segment!(s1::MVector{3,Float64}, s2::MVector{3,Float64}, α::Float64, τ::Float64, nsub::Int) -> nothing

Integrate a single MS segment of duration `τ` using `nsub` Strang substeps.

Inputs
- `(s1,s2)` are modified in place and are renormalized at each substep.

- Used in `ms_residual!` and `verify_nodes` to propagate from node `i` to node `i+1`.
"""
@inline function integrate_segment!(s1::MVector{3,Float64},s2::MVector{3,Float64},α::Float64,τ::Float64,nsub::Int)
    h=τ/nsub
    @inbounds for _ in 1:nsub
        strang_step!(s1,s2,h,α;renorm=true)
    end
    return nothing
end

############################
# Residual F(y): segment continuity
############################
"""
    ms_residual!(F::Vector{Float64}, y::Vector{Float64}, α::Float64, E::Float64, ϕ2sec::Float64; M::Int, nsub::Int) -> nothing

Compute the multiple-shooting residual vector `F(y)`.

Unknown vector
- `y` packs `M` node spins `(s1_i,s2_i)` plus the period `T`.

Residual structure (length `m = 6M + 2`)
1. Segment continuity (6M components):
   For each segment `i`, integrate node `i` forward by `τ = T/M` and compare
   to the next node `j = i+1` (cyclic):
   `Δ1 = Φ_τ(s1_i,s2_i).s1 - s1_j`, `Δ2 = Φ_τ(...).s2 - s2_j`,
   written componentwise into `F`.
2. Section constraint (1 component):
   `phase_g(s2_1, ϕ2sec) = 0`.
3. Energy constraint (1 component):
   `H(s1_1, s2_1, α) - E = 0`.

Comment
- The constraints are applied at node 1 to fix time-shift and energy. Energy is conserved since Strang is a symplectic integrator, so the energy constraint should be well-behaved.
"""
function ms_residual!(F::Vector{Float64},y::Vector{Float64},α::Float64,E::Float64,ϕ2sec::Float64;M::Int,nsub::Int)
    s1s,s2s,T=unpack(y,M)
    τ=T/M
    k=1
    s1w=MVector{3,Float64}(0,0,0)
    s2w=MVector{3,Float64}(0,0,0)
    @inbounds for i in 1:M
        s1w.=s1s[i];s2w.=s2s[i]
        integrate_segment!(s1w,s2w,α,τ,nsub)
        j=(i<M) ? (i+1) : 1
        Δ1=Vec3(s1w)-s1s[j]
        Δ2=Vec3(s2w)-s2s[j]
        F[k]=Δ1[1];k+=1;F[k]=Δ1[2];k+=1;F[k]=Δ1[3];k+=1
        F[k]=Δ2[1];k+=1;F[k]=Δ2[2];k+=1;F[k]=Δ2[3];k+=1
    end
    F[k]=phase_g(s2s[1],ϕ2sec);k+=1
    F[k]=H(s1s[1],s2s[1],α)-E;k+=1
    return nothing
end

############################
# Segment-only J*v and J'*w using As[i]  (continuity rows only)
############################
"""
    apply_Jseg!(out::Vector{Float64}, v::AbstractVector{Float64}, As::Vector{Matrix{Float64}}; M::Int) -> nothing

Apply the segment continuity Jacobian block to a vector: `out = J_seg * v`. Here `J_seg` corresponds only to the 6M continuity rows (no section/energy, no `T`).

Inputs
- `v` is the stacked node variation of length `6M`.
- `As[i]` is the 6×6 tangent map for segment `i`.

For each segment `i` (cyclic next index `ip1`):
- `out_i = As[i] * v_i - v_{ip1}` (each `out_i` is 6 components).

- Used in Matrix-free LM/GN to assemble `J*v` without forming a global Jacobian.
"""
function apply_Jseg!(out::Vector{Float64},v::AbstractVector{Float64},As::Vector{Matrix{Float64}};M::Int)
    @inbounds for i in 1:(6*M)
        out[i]=0.0
    end
    @inbounds for i in 1:M
        ip1=(i<M) ? (i+1) : 1
        bi=6*(i-1);bp=6*(ip1-1)
        A=As[i]
        for row in 1:6
            s=0.0
            for col in 1:6
                s+=A[row,col]*v[bi+col]
            end
            out[bi+row]=s - v[bp+row]
        end
    end
    return nothing
end
"""
    apply_Jtseg!(out::Vector{Float64}, w::AbstractVector{Float64}, As::Vector{Matrix{Float64}}; M::Int) -> nothing

Apply the transpose of the *segment continuity* Jacobian block: `out = J_seg' * w`.

Inputs
- `w` is a vector of length `6M` corresponding to continuity-row weights.
- `As[i]` is the 6×6 tangent map for segment `i`.

Operation (per segment `i`, cyclic next `ip1`)
- Accumulate `out_i += As[i]' * w_i`
- Accumulate `out_{ip1} -= w_i`

- Used in Matrix-free computation of `J' * w` in LM/GN (adjoint action).
"""
function apply_Jtseg!(out::Vector{Float64},w::AbstractVector{Float64},As::Vector{Matrix{Float64}};M::Int)
    @inbounds for i in 1:(6*M); out[i]=0.0; end
    @inbounds for i in 1:M
        ip1=(i<M) ? (i+1) : 1
        bi=6*(i-1);bp=6*(ip1-1)
        A=As[i]
        for col in 1:6
            s=0.0
            for row in 1:6
                s+=A[row,col]*w[bi+row]
            end
            out[bi+col]+=s
        end
        for row in 1:6
            out[bp+row]-=w[bi+row]
        end
    end
    return nothing
end

############################
# Constraint gradients at node1 (φ-section + energy)
############################
"""
    constraint_grads!(gφ::MVector{6,Float64}, gE::MVector{6,Float64}, y::Vector{Float64}, α::Float64) -> nothing

Compute gradients of the two scalar constraints at node 1 with respect to the
6 node-1 coordinates `(s1x,s1y,s1z,s2x,s2y,s2z)`.

Outputs
- `gφ`: gradient of the section constraint `phase_g(s2_1, ϕ2sec)` **without**
  the outer wrapping/constant shift; effectively gradient of `atan2(s2y,s2x)`.
  Nonzero only in `(s2x,s2y)` and set to zero near the pole `s2x=s2y=0`.
- `gE`: gradient of the energy constraint `H(s1_1,s2_1,α) - E`.

- Used in `ms_lm_mfree_strang` to augment `J*v` and `J'*w` with the last two rows.
"""
@inline function constraint_grads!(gφ::MVector{6,Float64},gE::MVector{6,Float64},y::Vector{Float64},α::Float64)
    s1=Vec3(y[1],y[2],y[3]);s2=Vec3(y[4],y[5],y[6])
    x=s2[1];yy=s2[2];r2=x*x+yy*yy
    if r2<1e-30
        gφ.=0.0
    else
        gφ[1]=0.0;gφ[2]=0.0;gφ[3]=0.0
        gφ[4]=-yy/r2;gφ[5]=x/r2;gφ[6]=0.0
    end
    gE[1]=α*s2[1];gE[2]=0.0;gE[3]=1.0
    gE[4]=α*s1[1];gE[5]=0.0;gE[6]=1.0
    return nothing
end

mutable struct CGWS
    r::Vector{Float64}
    p::Vector{Float64}
    Ap::Vector{Float64}
end
"""
    mutable struct CGWS
        r::Vector{Float64}
        p::Vector{Float64}
        Ap::Vector{Float64}
    end

Workspace for the conjugate-gradient solver `cg_solve!`.

Fields
- `r`: residual vector
- `p`: search direction
- `Ap`: temporary for matrix-vector product `A*p`

Constructors
- `CGWS(n::Int)` allocates zeroed buffers of length `n`.

- Used Inside matrix-free LM/GN to solve `(J'J + λI) δ = -J'F`.
"""
CGWS(n::Int)=CGWS(zeros(n),zeros(n),zeros(n))

"""
    cg_solve!(x::Vector{Float64}, applyA!::Function, b::Vector{Float64}, ws::CGWS; tol::Float64=1e-8, maxit::Int=200)
        -> (it::Int, rel::Float64)

Solve the linear system `A*x = b` using (unpreconditioned) conjugate gradients,
given a matrix-free operator `applyA!(out, v)` that computes `out = A*v`.

Inputs
- `x`: solution vector (overwritten; initialized to zeros).
- `applyA!`: function `(out, v) -> nothing`.
- `b`: right-hand side.
- `ws`: workspace (`CGWS`) providing `r`, `p`, `Ap`.

Stopping criterion
- Returns when `||r||/||r0|| < tol` or after `maxit` iterations.

Returns
- `it`: number of iterations performed.
- `rel`: final relative residual `sqrt(rs/rs0)`.
"""
function cg_solve!(x::Vector{Float64},applyA!::Function,b::Vector{Float64},ws::CGWS;tol::Float64=1e-8,maxit::Int=200)
    r=ws.r;p=ws.p;Ap=ws.Ap
    fill!(x,0.0)
    r.=b
    p.=r
    rs=dot(r,r);rs0=rs
    rs0==0.0 && return (it=0,rel=0.0)
    for it in 1:maxit
        applyA!(Ap,p)
        denom=dot(p,Ap)
        denom==0.0 && return (it=it,rel=sqrt(rs/rs0))
        α=rs/denom
        @inbounds for i in eachindex(x)
            x[i]+=α*p[i]
            r[i]-=α*Ap[i]
        end
        rsn=dot(r,r)
        rel=sqrt(rsn/rs0)
        rel<tol && return (it=it,rel=rel)
        β=rsn/rs
        @inbounds for i in eachindex(p)
            p[i]=r[i]+β*p[i]
        end
        rs=rsn
    end
    return (it=maxit,rel=sqrt(rs/rs0))
end

# --------------------------
# Matrix-free LM/GN for strang flow
# --------------------------
"""
    ms_lm_mfree_strang(y0::Vector{Float64}, α::Float64, E::Float64, ϕ2sec::Float64; M::Int, nsub::Int=3000, tol::Float64=1e-12, maxit::Int=50, λ0::Float64=1e-6, λmin::Float64=1e-14, λmax::Float64=1e16, cg_tol::Float64=1e-6, cg_maxit::Int=400, dT_eps::Float64=1e-7, verbose::Bool=true) -> (y::Vector{Float64}, status::Symbol, r::Float64)

Matrix-free Levenberg–Marquardt / Gauss–Newton solver for the multiple-shooting
system using the Strang flow.

Problem
- Minimize `||F(y)||` where `F(y)` is the MS residual from `ms_residual!`.

Key idea
- Avoid forming the full Jacobian `J` or `J'J`.
- Use per-segment tangent maps `As[i]` to implement matrix-vector products:
  - `J*v` via continuity action + finite-difference `dF/dT` + constraint rows
  - `J'*w` via adjoint continuity action + constraint gradients + `T` coupling

- Each LM step solves `(J'J + λI) δ = -J'F` with conjugate gradients (`cg_solve!`)
  using a matrix-free `applyA!`.

Returns
- `:converged` if `norm(F) < tol`
- `:stalled` if no improvement found within damping attempts
- `:maxit` if iteration limit reached
"""
function ms_lm_mfree_strang(y0::Vector{Float64},α::Float64,E::Float64,ϕ2sec::Float64;
    M::Int,nsub::Int=3000,
    tol::Float64=1e-12,maxit::Int=50,
    λ0::Float64=1e-6,λmin::Float64=1e-14,λmax::Float64=1e16,
    cg_tol::Float64=1e-6,cg_maxit::Int=400,
    dT_eps::Float64=1e-7,verbose::Bool=true)
    BLAS.set_num_threads(round(Int,Sys.CPU_THREADS/2)) # max blas threads for J'J matvecs
    y=copy(y0)
    n=6*M+1
    m=6*M+2
    length(y)==n || throw(DimensionMismatch("y length $(length(y)) expected $n"))
    As=[zeros(6,6) for _ in 1:M]
    F=zeros(m);Ft=zeros(m);Fp=zeros(m);Fm=zeros(m);dFdT=zeros(m)
    segtmp=zeros(6*M)
    Jtv=zeros(6*M)
    gφ=MVector{6,Float64}(0,0,0,0,0,0)
    gE=MVector{6,Float64}(0,0,0,0,0,0)
    δ=zeros(n);g=zeros(n);b=zeros(n)
    cgws=CGWS(n)
    tmpm=zeros(m)
    @inline function residual!(Fout::Vector{Float64},yy::Vector{Float64})
        ms_residual!(Fout,yy,α,E,ϕ2sec;M=M,nsub=nsub);return nothing
    end
    function update_dFdT!(yy::Vector{Float64})
        T=yy[end]
        eps=dT_eps*max(1.0,abs(T))
        yy[end]=T+eps;residual!(Fp,yy)
        yy[end]=T-eps;residual!(Fm,yy)
        yy[end]=T
        @inbounds for i in 1:m
            dFdT[i]=(Fp[i]-Fm[i])/(2*eps)
        end
        dFdT[6*M+1]=0.0
        dFdT[6*M+2]=0.0
        return nothing
    end
    function applyJ!(outm::Vector{Float64},v::Vector{Float64},yy::Vector{Float64})
        apply_Jseg!(segtmp,view(v,1:6*M),As;M=M)
        @inbounds for i in 1:(6*M)
            outm[i]=segtmp[i] + dFdT[i]*v[end]
        end
        constraint_grads!(gφ,gE,yy,α)
        sφ=0.0;sE=0.0
        @inbounds for k in 1:6
            sφ+=gφ[k]*v[k]
            sE+=gE[k]*v[k]
        end
        outm[6*M+1]=sφ
        outm[6*M+2]=sE
        return nothing
    end
    function applyJt!(outn::Vector{Float64},w::Vector{Float64},yy::Vector{Float64})
        apply_Jtseg!(Jtv,view(w,1:6*M),As;M=M)
        @inbounds for i in 1:(6*M)
            outn[i]=Jtv[i]
        end
        constraint_grads!(gφ,gE,yy,α)
        wφ=w[6*M+1];wE=w[6*M+2]
        @inbounds for k in 1:6
            outn[k]+=gφ[k]*wφ + gE[k]*wE
        end
        outn[end]=dot(view(dFdT,1:6*M),view(w,1:6*M))
        return nothing
    end
    function applyA!(outn::Vector{Float64},xv::Vector{Float64},yy::Vector{Float64},λ::Float64)
        applyJ!(tmpm,xv,yy)
        applyJt!(outn,tmpm,yy)
        @inbounds for i in 1:n
            outn[i]+=λ*xv[i]
        end
        return nothing
    end
    build_As_strang!(As,y,α;M=M,nsub=nsub)
    residual!(F,y)
    update_dFdT!(y)
    r=norm(F)
    λ=λ0
    verbose && println("ms_mfree it=0  r=$(r)  T=$(y[end])  λ=$(λ)  (m=$m,n=$n)")
    for it in 1:maxit
        r<tol && return y,:converged,r
        applyJt!(g,F,y)
        @inbounds for i in 1:n
            b[i]=-g[i]
        end
        function _applyA!(outv,pv)
            applyA!(outv,pv,y,λ);return nothing
        end
        cginfo=cg_solve!(δ,_applyA!,b,cgws;tol=cg_tol,maxit=cg_maxit)
        improved=false
        for _ in 1:30
            t=1.0
            while t>1e-12
                ytry=y .+ t.*δ
                @inbounds for i in 1:M
                    b0=6*(i-1)
                    renorm_spin_at!(ytry,b0+1)
                    renorm_spin_at!(ytry,b0+4)
                end
                ytry[end]=max(1e-12,ytry[end])
                nodes_in_domain(ytry,M) || (t*=0.5;continue)

                build_As_strang!(As,ytry,α;M=M,nsub=nsub)
                residual!(Ft,ytry)
                update_dFdT!(ytry)
                rt=norm(Ft)
                if rt<r
                    y.=ytry;F.=Ft;r=rt
                    λ=max(λ*0.5,λmin)
                    improved=true
                    break
                end
                t*=0.5
            end
            improved && break
            λ=min(λ*10,λmax)
        end
        verbose && println("ms_mfree it=$(it)  r=$(r)  T=$(y[end])  λ=$(λ)  cg_it=$(cginfo.it) cg_rel=$(cginfo.rel)")
        !improved && return y,:stalled,r
    end
    return y,:maxit,r
end

############################
# Initial seeding by forward flow
############################
"""
    spin_from_angles(θ::Float64, ϕ::Float64) -> Vec3

Construct a unit spin on the sphere from polar angle `θ ∈ [0,π]` and azimuth
`ϕ ∈ (-π,π]`.

Convention
- `(x,y,z) = (sinθ cosϕ, sinθ sinϕ, cosθ)`.
"""
@inline function spin_from_angles(θ::Float64,ϕ::Float64)
    sθ,cθ=sincos(θ);sϕ,cϕ=sincos(ϕ)
    return Vec3(sθ*cϕ,sθ*sϕ,cθ)
end
"""
    rand_spin!(rng::AbstractRNG) -> Vec3

Sample a random unit spin uniformly on the sphere using `rng`.

Returns
- `Vec3` on the unit sphere.
"""
@inline function rand_spin!(rng::AbstractRNG)
    u=rand(rng);v=rand(rng);z=2u-1;ϕ=_TWO_PI*v;θ=acos(clamp(z,-1.0,1.0))
    return spin_from_angles(θ,ϕ)
end
"""
    rand_spin_fixed_ϕ!(rng::AbstractRNG, ϕ::Float64) -> Vec3

Sample a random unit spin with azimuth fixed to `ϕ` and polar angle distributed
so that the resulting points are uniform in `z` (i.e. uniform in `cosθ`).

Returns
- `Vec3` on the unit sphere with azimuth `ϕ`.
"""
@inline function rand_spin_fixed_ϕ!(rng::AbstractRNG,ϕ::Float64)
    u=rand(rng)
    z=2u-1
    θ=acos(clamp(z,-1.0,1.0))
    return spin_from_angles(θ,ϕ)
end
"""
    enforce_ϕ!(s::Vec3, ϕ::Float64) -> Vec3

Adjust the azimuth of an existing spin `s` to be exactly `ϕ`, preserving:
- its z-component `s_z`
- its transverse radius `r = hypot(s_x, s_y)` (up to renormalization)

If `r` is extremely small (near the pole), returns `s` unchanged.

- Used when seeding initial conditions so that `s2` starts on the chosen section.
"""
@inline function enforce_ϕ!(s::Vec3,ϕ::Float64)
    r=hypot(s[1],s[2])
    r<1e-14 && return s
    sϕ,cϕ=sincos(ϕ)
    return normalize3(Vec3(r*cϕ,r*sϕ,s[3]))
end
"""
    ms_seed_from_flow(s1_0::Vec3, s2_0::Vec3, α::Float64, T0::Float64, M::Int; nsub_total::Int=20000) -> Vector{Float64}

Create an initial multiple-shooting guess `y0` by forward integrating the Strang
flow over a trial period `T0` and sampling `M` equally spaced nodes.

- Integrate from `(s1_0,s2_0)` with step `h = T0/nsub_total`.
- Whenever time crosses `k*T0/M`, record the current normalized spins as node `k`.
- Pack nodes and `T0` into `y0` using `pack!`.

Returns
- `y0` of length `6M+1`.
"""
function ms_seed_from_flow(s1_0::Vec3,s2_0::Vec3,α::Float64,T0::Float64,M::Int;nsub_total::Int=20000)
    s1s=Vector{Vec3}(undef,M)
    s2s=Vector{Vec3}(undef,M)
    s1=MVector{3,Float64}(s1_0)
    s2=MVector{3,Float64}(s2_0)
    t=0.0;k=1
    s1s[1]=normalize3(Vec3(s1))
    s2s[1]=normalize3(Vec3(s2))
    next_t=T0/M
    h=T0/nsub_total
    @inbounds for _ in 1:nsub_total
        strang_step!(s1,s2,h,α;renorm=true)
        t+=h
        if t>=next_t && k<M
            k+=1
            s1s[k]=normalize3(Vec3(s1))
            s2s[k]=normalize3(Vec3(s2))
            next_t=k*T0/M
        end
    end
    y0=zeros(Float64,6*M+1)
    pack!(y0,s1s,s2s,T0)
    return y0
end

############################
# Find one PO using FD Jacobian and strang symplectic 
############################
"""
    find_po_ms_lm_once(α::Float64, E::Float64, ϕ2sec::Float64, T0::Float64; M::Int=150, nsub_seed_total::Int=12000, rng::AbstractRNG=Random.MersenneTwister(1), nsub_screen::Int=350, tol_screen::Float64=1e-6, maxit_screen::Int=10, λ0_screen::Float64=1e-4, nsub_full::Int=3000, tol_full::Float64=1e-12, maxit_full::Int=60,λ0_full::Float64=1e-6, verbose_screen::Bool=false, verbose_full::Bool=false) -> NamedTuple

Attempt to find a single periodic orbit (PO) satisfying:
- segment continuity over `M` nodes,
- section constraint `ϕ2 = ϕ2sec` at node 1,
- energy constraint `H = E` at node 1,

using the matrix-free Strang LM/GN solver.

1. Random seed: draw `s1_0` uniformly; draw `s2_0` with fixed azimuth `ϕ2sec`.
2. Build initial MS guess `y0` by forward flow (`ms_seed_from_flow`).
3. "Screen" solve: coarse integration (`nsub_screen`) + loose tolerance.
4. Full solve: fine integration (`nsub_full`) + tight tolerance.

Returns
- `(y, status, rr, T)` where `rr = norm(F(y))` from the final stage.
"""
function find_po_ms_lm_once(α::Float64,E::Float64,ϕ2sec::Float64,T0::Float64;
    M::Int=150,
    nsub_seed_total::Int=12000,rng::AbstractRNG=Random.MersenneTwister(1),
    nsub_screen::Int=350,tol_screen::Float64=1e-6,maxit_screen::Int=10,λ0_screen::Float64=1e-4,
    nsub_full::Int=3000,tol_full::Float64=1e-12,maxit_full::Int=60,λ0_full::Float64=1e-6,
    verbose_screen::Bool=false,verbose_full::Bool=false)
    s1_0=rand_spin!(rng)
    s2_0=enforce_ϕ!(rand_spin_fixed_ϕ!(rng,ϕ2sec),ϕ2sec)
    y0=ms_seed_from_flow(s1_0,s2_0,α,T0,M;nsub_total=nsub_seed_total)
    y1,st1,rr1=ms_lm_mfree_strang(y0,α,E,ϕ2sec;M=M,nsub=nsub_screen,tol=tol_screen,maxit=maxit_screen,λ0=λ0_screen,cg_tol=1e-4,cg_maxit=200,verbose=verbose_screen)
    if !(st1==:converged || rr1<1e-2)
        return (y=y1,status=:rejected,rr=rr1,T=y1[end])
    end
    y2,st2,rr2=ms_lm_mfree_strang(y1,α,E,ϕ2sec;M=M,nsub=nsub_full,tol=tol_full,maxit=maxit_full,λ0=λ0_full,cg_tol=1e-6,cg_maxit=400,verbose=verbose_full)
    return (y=y2,status=st2,rr=rr2,T=y2[end])
end

############################
# Verify MS nodes (continuity + energy at nodes)
############################
"""
    verify_nodes(y::Vector{Float64}, α::Float64, E::Float64; M::Int, nsub::Int=3000) -> NamedTuple

Post-solution verification for a multiple-shooting orbit.

Checks
- For each node `i`, compute energy `H(s1_i,s2_i,α)`:
  - track `Emin`, `Emax`, and maximum absolute deviation from target `E`.
- For each segment, integrate node `i` forward by `τ = T/M` and measure
  continuity defects to node `i+1`:
  - `dmax1`, `dmax2`, and `dmax = max(dmax1,dmax2)`.

Returns
- `(T, dmax1, dmax2, dmax, Emin, Emax, Espan, max_abs_H_err)`.
"""
function verify_nodes(y::Vector{Float64},α::Float64,E::Float64;M::Int,nsub::Int=3000)
    s1s,s2s,T=unpack(y,M);τ=T/M
    s1w=MVector{3,Float64}(0,0,0);s2w=MVector{3,Float64}(0,0,0)
    dmax1=0.0;dmax2=0.0
    Emin=Inf;Emax=-Inf
    max_abs_H_err=0.0
    @inbounds for i in 1:M
        Ei=H(s1s[i],s2s[i],α)
        Ei<Emin && (Emin=Ei);Ei>Emax && (Emax=Ei)
        eerr=abs(Ei-E);eerr>max_abs_H_err && (max_abs_H_err=eerr)
        s1w.=s1s[i];s2w.=s2s[i]
        integrate_segment!(s1w,s2w,α,τ,nsub)
        j=(i<M) ? (i+1) : 1
        d1=norm(Vec3(s1w)-s1s[j])
        d2=norm(Vec3(s2w)-s2s[j])
        d1>dmax1 && (dmax1=d1)
        d2>dmax2 && (dmax2=d2)
    end
    return (T=T,dmax1=dmax1,dmax2=dmax2,dmax=max(dmax1,dmax2),
            Emin=Emin,Emax=Emax,Espan=Emax-Emin,max_abs_H_err=max_abs_H_err)
end

############################
# Sample one period uniformly and save
############################
"""
    sample_one_period(y::Vector{Float64}, α::Float64; M::Int, Nsave::Int=20000, renorm::Bool=true) -> NamedTuple

Sample the periodic orbit over one full period using uniform time steps.

Inputs
- `y`: MS solution (nodes + period).
- Starts from node 1 `(s1_1,s2_1)`.
- Uses `dt = T/Nsave` and advances with `strang_step!`.

Returns
- `(t, s1, s2, T, dt)` where `t` has length `Nsave+1` and includes both endpoints.
"""
function sample_one_period(y::Vector{Float64},α::Float64;M::Int,Nsave::Int=20000,renorm::Bool=true)
    s1s,s2s,T=unpack(y,M)
    s1=MVector{3,Float64}(s1s[1]);s2=MVector{3,Float64}(s2s[1])
    t=Vector{Float64}(undef,Nsave+1)
    S1=Vector{Vec3}(undef,Nsave+1)
    S2=Vector{Vec3}(undef,Nsave+1)
    t[1]=0.0;S1[1]=Vec3(s1);S2[1]=Vec3(s2)
    dt=T/Nsave
    @inbounds for k in 1:Nsave
        strang_step!(s1,s2,dt,α;renorm=renorm)
        t[k+1]=k*dt
        S1[k+1]=Vec3(s1);S2[k+1]=Vec3(s2)
    end
    return (t=t,s1=S1,s2=S2,T=T,dt=dt)
end

@inline _fmt_g(digits::Int)=Printf.Format("%."*string(digits)*"g")
function write_traj_csv(path::String,meta::Dict{String,Any},t::Vector{Float64},s1::Vector{Vec3},s2::Vector{Vec3};digits::Int=16)
    F=_fmt_g(digits)
    open(path,"w") do io
        for (k,v) in meta; println(io,"# ",k,"=",v); end
        println(io,"t,s1x,s1y,s1z,s2x,s2y,s2z")
        @inbounds for i in eachindex(t)
            print(io,Printf.format(F,t[i]));print(io,",")
            print(io,Printf.format(F,s1[i][1]));print(io,",")
            print(io,Printf.format(F,s1[i][2]));print(io,",")
            print(io,Printf.format(F,s1[i][3]));print(io,",")
            print(io,Printf.format(F,s2[i][1]));print(io,",")
            print(io,Printf.format(F,s2[i][2]));print(io,",")
            print(io,Printf.format(F,s2[i][3]));print(io,"\n")
        end
    end
    return nothing
end
function write_nodes_csv(path::String,meta::Dict{String,Any},y::Vector{Float64};M::Int,digits::Int=16)
    F=_fmt_g(digits)
    s1s,s2s,T=unpack(y,M)
    open(path,"w") do io
        for (k,v) in meta; println(io,"# ",k,"=",v); end
        println(io,"node,s1x,s1y,s1z,s2x,s2y,s2z")
        @inbounds for i in 1:M
            print(io,i);print(io,",")
            print(io,Printf.format(F,s1s[i][1]));print(io,",")
            print(io,Printf.format(F,s1s[i][2]));print(io,",")
            print(io,Printf.format(F,s1s[i][3]));print(io,",")
            print(io,Printf.format(F,s2s[i][1]));print(io,",")
            print(io,Printf.format(F,s2s[i][2]));print(io,",")
            print(io,Printf.format(F,s2s[i][3]));print(io,"\n")
        end
        println(io,"# T=",Printf.format(F,T))
    end
    return nothing
end
function save_figures(prefix::String,t::Vector{Float64},s1::Vector{Vec3},s2::Vector{Vec3};dpi=200)
    # xy
    fig=Figure(size=(900,350))
    ax1=Axis(fig[1,1],xlabel="s1x",ylabel="s1y",title="Spin 1 (x,y)")
    ax2=Axis(fig[1,2],xlabel="s2x",ylabel="s2y",title="Spin 2 (x,y)")
    lines!(ax1,[v[1] for v in s1],[v[2] for v in s1])
    lines!(ax2,[v[1] for v in s2],[v[2] for v in s2])
    save(prefix*"_xy.png",fig;px_per_unit=dpi/100)
    # cylinder 
    fig=Figure(size=(900,420))
    ax=Axis3(fig[1,1],xlabel="cosϕ",ylabel="sinϕ",zlabel="z",title="Cylinder embeddings")
    ϕ1=[φ_of(v) for v in s1]; ϕ2=[φ_of(v) for v in s2]
    lines!(ax,cos.(ϕ1),sin.(ϕ1),[v[3] for v in s1])
    lines!(ax,cos.(ϕ2),sin.(ϕ2),[v[3] for v in s2])
    save(prefix*"_cyl.png",fig;px_per_unit=dpi/100)
    return nothing
end

############################
# De-dup compare (T, node1)
# ##########################
"""
    is_duplicate(y::Vector{Float64}, ys::Vector{Vector{Float64}}; M::Int,
                 tol_T::Float64=1e-7, tol_node::Float64=1e-6) -> Bool

Heuristic de-duplication test for periodic-orbit solutions.

Two solutions are considered duplicates if:
1. Their periods satisfy `|T - Tb| ≤ tol_T`, and
2. Their node-1 spins are close: - `||s1_1 - s1b_1|| ≤ tol_node` and `||s2_1 - s2b_1|| ≤ tol_node`.

- Hopefully avoids saving multiple converged MS solutions that correspond to the same orbit.
"""
function is_duplicate(y::Vector{Float64},ys::Vector{Vector{Float64}};M::Int,tol_T::Float64=1e-7,tol_node::Float64=1e-6)
    s1s,s2s,T=unpack(y,M);s10=s1s[1];s20=s2s[1]
    for yb in ys
        s1b,s2b,Tb=unpack(yb,M)
        if abs(T-Tb)<=tol_T
            if norm(s10-s1b[1])<=tol_node && norm(s20-s2b[1])<=tol_node
                return true
            end
        end
    end
    return false
end

############################
# Collect many POs for a given α
############################
"""
    collect_pos(α::Float64, E::Float64, ϕ2sec::Float64; M::Int=150, target::Int=15, max_trials::Int=400, T0s::AbstractVector{Float64}=(2*pi).*collect(range(0.6,1.6,length=7)), nsub_seed_total::Int=12000, nsub_screen::Int=350, tol_screen::Float64=1e-6, maxit_screen::Int=10, λ0_screen::Float64=1e-4, nsub_full::Int=3000,  tol_full::Float64=1e-12, maxit_full::Int=60, λ0_full::Float64=1e-6, rng::AbstractRNG=Random.MersenneTwister(123), outdir::String="PO_DB", Nsave::Int=20000, dup_tol_T::Float64=1e-7, dup_tol_node::Float64=1e-6, digits::Int=16, verbose::Bool=true) -> NamedTuple

Collect a database of distinct periodic orbits for fixed parameters `(α,E,ϕ2sec)`.

- Repeats up to `max_trials` attempts until `target` unique converged solutions
  are found:
  1. Choose a trial period `T0` cycling through `T0s`.
  2. Call `find_po_ms_lm_once` to attempt convergence.
  3. Reject non-converged or duplicate solutions (`is_duplicate`).
  4. Verify (`verify_nodes`) and then save:
     - dense trajectory CSV (`write_traj_csv`)
     - node CSV (`write_nodes_csv`)
     - diagnostic figures (`save_figures`)

Filesystem layout
- Creates `outdir/alpha_<...>_E_<...>_phi2_<...>_strang_mfree/`
- Each accepted orbit uses prefix `po_###`.

Returns
- `(ys, rr, dir)` where:
  - `ys` is a vector of packed solutions `y`,
  - `rr` is the list of residual norms,
  - `dir` is the output directory path.
"""
function collect_pos(α::Float64,E::Float64,ϕ2sec::Float64;M::Int=150,target::Int=15,max_trials::Int=400,T0s::AbstractVector{Float64}=(2*pi).*collect(range(0.6,1.6,length=7)),nsub_seed_total::Int=12000,nsub_screen::Int=350,tol_screen::Float64=1e-6,maxit_screen::Int=10,λ0_screen::Float64=1e-4,nsub_full::Int=3000,tol_full::Float64=1e-12,maxit_full::Int=60,λ0_full::Float64=1e-6,rng::AbstractRNG=Random.MersenneTwister(123),outdir::String="PO_DB",Nsave::Int=20000,dup_tol_T::Float64=1e-7,dup_tol_node::Float64=1e-6,digits::Int=16,verbose::Bool=true)

    isdir(outdir) || mkpath(outdir)
    subdir=joinpath(outdir,@sprintf("alpha_%.6g_E_%.6g_phi2_%.6g_strang_mfree",α,E,ϕ2sec))
    isdir(subdir) || mkpath(subdir)
    accepted=Vector{Vector{Float64}}()
    rr_list=Float64[]
    trial=0
    while length(accepted)<target && trial<max_trials
        println("=== Trial $(trial+1) / $max_trials  (found $(length(accepted)) / $target) ===")
        trial+=1
        T0=T0s[1+mod(trial-1,length(T0s))]
        rep=find_po_ms_lm_once(α,E,ϕ2sec,T0;M=M,nsub_seed_total=nsub_seed_total,rng=rng,
            nsub_screen=nsub_screen,tol_screen=tol_screen,maxit_screen=maxit_screen,λ0_screen=λ0_screen,
            nsub_full=nsub_full,tol_full=tol_full,maxit_full=maxit_full,λ0_full=λ0_full,
            verbose_screen=false,verbose_full=false)
        if rep.status!=:converged
            verbose && println("trial=$trial  T0=$T0  status=$(rep.status)  rr=$(rep.rr)")
            continue
        end
        y=rep.y
        if is_duplicate(y,accepted;M=M,tol_T=dup_tol_T,tol_node=dup_tol_node)
            verbose && println("trial=$trial  converged but DUPLICATE (T=$(rep.T), rr=$(rep.rr))")
            continue
        end
        ver=verify_nodes(y,α,E;M=M,nsub=nsub_full)
        println("trial=$trial  converged OK (T=$(rep.T), rr=$(rep.rr))  verify: dmax=$(ver.dmax), max_abs_H_err=$(ver.max_abs_H_err)")
        push!(accepted,y)
        push!(rr_list,rep.rr)
        idx=length(accepted)
        prefix=joinpath(subdir,@sprintf("po_%03d",idx))
        sam=sample_one_period(y,α;M=M,Nsave=Nsave,renorm=true)
        meta=Dict{String,Any}("alpha"=>α,"E"=>E,"phi2sec"=>ϕ2sec,"flow"=>"strang_mfree","rr"=>rep.rr,"T"=>sam.T,"dt"=>sam.dt,"N"=>Nsave)
        write_traj_csv(prefix*"_traj.csv",meta,sam.t,sam.s1,sam.s2;digits=digits)
        write_nodes_csv(prefix*"_nodes.csv",meta,y;M=M,digits=digits)
        save_figures(prefix,sam.t,sam.s1,sam.s2)

        verbose && println("ACCEPT idx=$idx  rr=$(rep.rr)  T=$(sam.T)  saved -> $(prefix)_*")
    end
    verbose && println("DONE: accepted=$(length(accepted)) / target=$target (trials=$trial). Saved in: $subdir")
    return (ys=accepted,rr=rr_list,dir=subdir)
end
"""
    test_J_consistency(; α=2.2, E=0.0, ϕ2sec=pi/2, M=8, nsub=80, eps=1e-7, seed=1) -> NamedTuple

Numerical consistency test for the matrix-free Jacobian actions used in LM/GN.

Tests performed
1. Directional derivative check:
   - Compare `J*v` (implemented via `applyJ!`) against finite differences of `F(y ± eps*v)`.
2. Adjoint (transpose) test:
   - Check `⟨J*v, w⟩ ≈ ⟨v, J'*w⟩` for random vectors `v` and `w`.

Returns
- `(rel_err, adj_err)` where:
  - `rel_err` is the relative error of the FD directional derivative,
  - `adj_err` is the relative adjoint mismatch.

Purpose
- Provides confidence that the matvecs match the true Jacobian of `ms_residual!`.
"""
function test_J_consistency(;α=2.2,E=0.0,ϕ2sec=pi/2,M=8,nsub=80,eps=1e-7,seed=1)
    rng=MersenneTwister(seed)
    s1s=[rand_spin!(rng) for _ in 1:M] # make a random y on the sphere
    s2s=[enforce_ϕ!(rand_spin_fixed_ϕ!(rng,ϕ2sec),ϕ2sec) for _ in 1:M]
    T0=2π*(1.2+0.2*rand(rng))
    y=zeros(6*M+1)
    pack!(y,s1s,s2s,T0)
    # build As at y
    As=[zeros(6,6) for _ in 1:M]
    build_As_strang!(As,y,α;M=M,nsub=nsub)
    m=6*M+2; n=6*M+1
    F0=zeros(m); Fp=zeros(m); Fm=zeros(m)
    ms_residual!(F0,y,α,E,ϕ2sec;M=M,nsub=nsub)
    # dFdT by FD 
    dFdT=zeros(m)
    ytmp=copy(y)
    T=y[end]; dT=1e-7*max(1.0,abs(T))
    ytmp[end]=T+dT; ms_residual!(Fp,ytmp,α,E,ϕ2sec;M=M,nsub=nsub)
    ytmp[end]=T-dT; ms_residual!(Fm,ytmp,α,E,ϕ2sec;M=M,nsub=nsub)
    ytmp[end]=T
    @inbounds for i in 1:m
        dFdT[i]=(Fp[i]-Fm[i])/(2dT)
    end
    dFdT[6*M+1]=0.0
    dFdT[6*M+2]=0.0
    segtmp=zeros(6*M)
    Jtv=zeros(6*M)
    gφ=MVector{6,Float64}(0,0,0,0,0,0)
    gE=MVector{6,Float64}(0,0,0,0,0,0)
    function applyJ!(outm,v,yy)
        apply_Jseg!(segtmp,view(v,1:6*M),As;M=M)
        @inbounds for i in 1:(6*M)
            outm[i]=segtmp[i] + dFdT[i]*v[end]
        end
        constraint_grads!(gφ,gE,yy,α)
        sφ=0.0
        sE=0.0
        @inbounds for k in 1:6
            sφ+=gφ[k]*v[k]
            sE+=gE[k]*v[k]
        end
        outm[6*M+1]=sφ
        outm[6*M+2]=sE
        return nothing
    end
    function applyJt!(outn,w,yy)
        apply_Jtseg!(Jtv,view(w,1:6*M),As;M=M)
        @inbounds for i in 1:(6*M)
            outn[i]=Jtv[i]
        end
        constraint_grads!(gφ,gE,yy,α)
        wφ=w[6*M+1]; wE=w[6*M+2]
        @inbounds for k in 1:6
            outn[k]+=gφ[k]*wφ + gE[k]*wE
        end
        outn[end]=dot(view(dFdT,1:6*M),view(w,1:6*M))
        return nothing
    end
    # random direction v (renormalize nodes so we perturb tangentially-ish)
    v=randn(rng,n)
    # make node parts small so we don't destroy sphere too much
    v[1:6*M].*=1e-2
    v[end]*=1e-2
    # finite-diff directional derivative
    yplus=y.+eps.*v
    yminus=y.-eps.*v
    yplus[end]=max(1e-12,yplus[end]);yminus[end]=max(1e-12,yminus[end])
    Fplus=zeros(m); Fminus=zeros(m)
    ms_residual!(Fplus,yplus,α,E,ϕ2sec;M=M,nsub=nsub)
    ms_residual!(Fminus,yminus,α,E,ϕ2sec;M=M,nsub=nsub)
    fd=(Fplus.-Fminus)./(2*eps)
    Jv=zeros(m)
    applyJ!(Jv,v,y)
    rel_err=norm(fd-Jv)/max(norm(fd),1e-30)
    # adjoint test
    w=randn(rng,m)
    Jtw=zeros(n)
    applyJt!(Jtw,w,y)
    lhs=dot(Jv,w)
    rhs=dot(v,Jtw)
    adj_err = abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-30)
    println("Jv finite-diff rel err = ",rel_err)
    println("Adjoint test rel err  = ",adj_err)
    return (rel_err=rel_err, adj_err=adj_err)
end








##########################
# MAIN
##########################
# test J consistency to make sure the matvecs are the same as FD Jacobian. This is a key to trusting the LM/GN solver.
println("Testing J consistency...")
test_J_consistency(α=2.5,E=0.0,ϕ2sec=pi/2,M=8,nsub=80,eps=1e-6,seed=42)
# αs are λs in the Quantum Hamiltonian
αs=collect(range(2.0,4.0,step=0.1)) # best in the α>2.0 regime since we are interested in UPOs. For small α it likes to converge to a point (0,0) which is not useful. For the completely regular regime α<0.6 would not recommend this method at all.
#TODO penalize convergence to the origin
for α in αs
    E=0.0 # the energy of the PO.
    ϕ2sec=pi/2 # the section we want the PO to pierce (ϕ2=pi/2 means s2 is in the y-z plane)
    M=150 # number of MS nodes (segments)
    rng=Random.MersenneTwister(123)
    # max_trial - how many random initial seeds to try before giving up on finding target number of POs. This is a tradeoff between runtime and how many POs we find.
    # T0s - the trial periods to use for seeding. We cycle through these as we try different random seeds as to try to get in a basin of a PO. The final PO will probably have wildy different T
    # nsub_seed_total - how long to integrate when building the initial seed by forward flow. Longer is more expensive but offer better accuracy
    res=collect_pos(α,E,ϕ2sec;M=M,target=15,max_trials=500,T0s=0.5*pi*(1.5.+(10.0-1.5).*rand(rng,50)),nsub_seed_total=12000,outdir="PO",Nsave=20000,digits=16,verbose=true)
    println("Saved directory: ",res.dir)
end