import Base: +, -, *, /, ^, show, promote_rule, convert,zero

struct DualNumber{T<:Real} <: Number
    real::T
    dual::T
end

DualNumber(r::Real) = DualNumber(r, zero(r))
zero(::Type{DualNumber{T}}) where {T} = DualNumber(zero(T), zero(T))
zero(x::DualNumber) = DualNumber(zero(x.real), zero(x.dual))
Base.:-(x::DualNumber) = DualNumber(-x.real, -x.dual)
Base.:+(x::DualNumber) = x

convert(::Type{DualNumber{T}}, x::DualNumber) where {T} = DualNumber(convert(T, x.real), convert(T, x.dual))
convert(::Type{DualNumber{T}}, x::Real) where {T} = DualNumber(convert(T, x), zero(T))
promote_rule(::Type{DualNumber{T}}, ::Type{S}) where {T<:Real, S<:Real} = DualNumber{promote_type(T, S)}

+(x::DualNumber, y::DualNumber) = DualNumber(x.real + y.real, x.dual + y.dual)
-(x::DualNumber, y::DualNumber) = DualNumber(x.real - y.real, x.dual - y.dual)
*(x::DualNumber, y::DualNumber) = DualNumber(x.real * y.real, x.real * y.dual + x.dual * y.real)
/(x::DualNumber, y::DualNumber) = DualNumber(x.real / y.real, x.dual / y.real - (x.real * y.dual) / (y.real^2))
^(x::DualNumber, n::Integer) = n == 0 ? DualNumber(one(x.real), zero(x.real)) : DualNumber(x.real^n, n * x.real^(n-1) * x.dual)

Base.show(io::IO, x::DualNumber) = print(io, "$(x.real) + $(x.dual)ε")

Base.sin(x::DualNumber) = DualNumber(sin(x.real), cos(x.real) * x.dual)
Base.cos(x::DualNumber) = DualNumber(cos(x.real), -sin(x.real) * x.dual)
Base.exp(x::DualNumber) = DualNumber(exp(x.real), exp(x.real) * x.dual)
Base.log(x::DualNumber) = DualNumber(log(x.real), x.dual / x.real)

