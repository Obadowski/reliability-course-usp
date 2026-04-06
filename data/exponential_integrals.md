# Integrals of Exponential Functions — Worked Summary

Source page used for the list of targets: the Wikipedia article **“List of integrals of exponential functions”**.  
Primary source consulted in this session: https://en.wikipedia.org/wiki/List_of_integrals_of_exponential_functions

This note reorganizes the formulas into a cleaner markdown reference.  
Unless stated otherwise, every indefinite integral below is understood **up to an additive constant** `+ C`.

---

## 1. Integrals of polynomials times exponentials

### 1.1 Linear polynomial

$$
\int x e^{cx}\,dx
= e^{cx}\left(\frac{cx-1}{c^2}\right) + C,
\qquad c\neq 0
$$

### 1.2 Quadratic polynomial
$$
\int x^2 e^{cx}\,dx
= e^{cx}\left(\frac{x^2}{c}-\frac{2x}{c^2}+\frac{2}{c^3}\right)+C,
\qquad c\neq 0
$$

### 1.3 General polynomial

A useful reduction formula is

$$
\int x^n e^{cx}\,dx =
\frac{1}{c}x^n e^{cx}
-\frac{n}{c}\int x^{n-1}e^{cx}\,dx,
\qquad c\neq 0
$$

A closed form is

$$
\int x^n e^{cx}\,dx =
e^{cx}\sum_{i=0}^{n}(-1)^i
\frac{n!}{(n-i)!\,c^{i+1}}x^{\,n-i}+C
$$

Equivalent reindexing:

$$
\int x^n e^{cx}\,dx =
e^{cx}\sum_{i=0}^{n}(-1)^{n-i}
\frac{n!}{i!\,c^{\,n-i+1}}x^i + C
$$

### 1.4 Exponential over \(x\)

The page gives the series form

$$
\int \frac{e^{cx}}{x}\,dx=
\ln|x|+\sum_{n=1}^{\infty}\frac{(cx)^n}{n\,n!}+C
$$

A more standard special-function form is
\[
\int \frac{e^{cx}}{x}\,dx = \operatorname{Ei}(cx)+C
\]
because
\[
\operatorname{Ei}(z)=\gamma+\ln|z|+\sum_{n=1}^{\infty}\frac{z^n}{n\,n!}
\]
up to a constant convention.

### 1.5 Reduction for \(e^{cx}/x^n\)
For $$n\neq 1$$,

$$
\int \frac{e^{cx}}{x^n}\,dx=
\frac{1}{n-1}\left(
-\frac{e^{cx}}{x^{n-1}}
+
c\int \frac{e^{cx}}{x^{n-1}}\,dx
\right)+C
$$

This is a reduction identity rather than a final elementary closed form.

---

## 2. Integrals involving only exponential functions

### 2.1 Chain-rule form
\[
\int f'(x)e^{f(x)}\,dx = e^{f(x)} + C
\]

### 2.2 Pure exponential
\[
\int e^{cx}\,dx = \frac{1}{c}e^{cx}+C,
\qquad c\neq 0
\]

### 2.3 Exponential with arbitrary positive base
\[
\int a^x\,dx = \frac{a^x}{\ln a}+C,
\qquad a>0,\ a\neq 1
\]

---

## 3. Integrals involving the error function or exponential integral

### 3.1 \(e^{cx}\ln x\)

$$
\int e^{cx}\ln x\,dx=
\frac{1}{c}\left(e^{cx}\ln|x|-\operatorname{Ei}(cx)\right)+C,
\qquad c\neq 0
$$

### 3.2 \(x e^{cx^2}\)
\[
\int x e^{cx^2}\,dx = \frac{1}{2c}e^{cx^2}+C,
\qquad c\neq 0
\]

### 3.3 Gaussian-type integral

$$
\int e^{-cx^2}\,dx =
\sqrt{\frac{\pi}{4c}}\,
\operatorname{erf}(\sqrt{c}\,x)+C =
\frac{\sqrt{\pi}}{2\sqrt{c}}\operatorname{erf}(\sqrt{c}\,x)+C,
\qquad c>0
$$

If \(c<0\), the antiderivative is expressed with the imaginary error function \(\operatorname{erfi}\).

### 3.4 \(x e^{-cx^2}\)

$$
\int x e^{-cx^2}\,dx=
-\frac{1}{2c}e^{-cx^2}+C,
\qquad c\neq 0
$$

### 3.5 \(\dfrac{e^{-x^2}}{x^2}\)

$$
\int \frac{e^{-x^2}}{x^2}\,dx=
-\frac{e^{-x^2}}{x}
-\sqrt{\pi}\operatorname{erf}(x)+C
$$

### 3.6 Normal density

$$
\int
\frac{1}{\sigma\sqrt{2\pi}}
e^{-\frac12\left(\frac{x-\mu}{\sigma}\right)^2}\,dx=
\frac12\operatorname{erf}\!\left(
\frac{x-\mu}{\sigma\sqrt{2}}
\right)+C
$$

Strictly speaking, many textbooks would instead write the CDF in terms of

$$
\Phi\!\left(\frac{x-\mu}{\sigma}\right)=
\frac12\left[
1+\operatorname{erf}\!\left(
\frac{x-\mu}{\sigma\sqrt{2}}
\right)
\right]
$$

so the missing \(1/2\) is absorbed into the integration constant.

---

## 4. Other integrals

### 4.1 Recursive expansion for \(\int e^{x^2}\,dx\)

This integral is **not elementary**. The page gives the recursion

$$
\int e^{x^2}\,dx=
e^{x^2}\left(\sum_{j=0}^{n-1}c_{2j}\frac{1}{x^{2j+1}}\right)
+
(2n-1)c_{2n-2}\int \frac{e^{x^2}}{x^{2n}}\,dx
\qquad (n>0)
$$

with

$$
c_{2j}=
\frac{1\cdot 3\cdot 5\cdots (2j-1)}{2^{j+1}}
=
\frac{(2j)!}{j!\,2^{2j+1}}
$$

A standard special-function form is

$$
\int e^{x^2}\,dx=
\frac{\sqrt{\pi}}{2}\operatorname{erfi}(x)+C
$$

### 4.2 Power tower integral
The page includes the nontrivial identity

$$
\int \underbrace{x^{x^{\cdot^{\cdot^x}}}}_{m}\,dx=
\sum_{n=0}^{m}
\frac{(-1)^n (n+1)^{n-1}}{n!}\Gamma(n+1,-\ln x)
+
\sum_{n=m+1}^{\infty}(-1)^n a_{mn}\Gamma(n+1,-\ln x)
+C,
\qquad x>0
$$

where

$$
a_{mn}=
\begin{cases}
1, & n=0,\\[4pt]
\dfrac{1}{n!}, & m=1,\\[6pt]
\dfrac{1}{n}\sum_{j=1}^{n} j\,a_{m,n-j}a_{m-1,j-1}, & \text{otherwise}
\end{cases}
$$

and \(\Gamma(s,z)\) is the upper incomplete gamma function.

This is included for completeness. It is not a practical classroom formula.

### 4.3 Reciprocal of an affine exponential

$$
\int \frac{1}{ae^{\lambda x}+b}\,dx=
\frac{x}{b} -
\frac{1}{b\lambda}\ln(ae^{\lambda x}+b)+C
$$

Conditions stated on the page:
- \(b\neq 0\)
- \(\lambda\neq 0\)
- \(ae^{\lambda x}+b>0\) if one wants the real logarithm without absolute values.

A more robust real-valued form is

$$
\int \frac{1}{ae^{\lambda x}+b}\,dx =
\frac{x}{b}-
\frac{1}{b\lambda}\ln|ae^{\lambda x}+b|+C
$$

### 4.4 \(\dfrac{e^{2\lambda x}}{ae^{\lambda x}+b}\)

$$
\int \frac{e^{2\lambda x}}{ae^{\lambda x}+b}\,dx=
\frac{1}{a^2\lambda}
\left[
ae^{\lambda x}+b-b\ln(ae^{\lambda x}+b)
\right]+C
$$

Conditions stated on the page:
- \(a\neq 0\)
- \(\lambda\neq 0\)
- \(ae^{\lambda x}+b>0\) for the real logarithm as written.

Again, with absolute values:

$$
\int \frac{e^{2\lambda x}}{ae^{\lambda x}+b}\,dx=
\frac{1}{a^2\lambda}
\left[
ae^{\lambda x}+b-b\ln|ae^{\lambda x}+b|
\right]+C
$$

### 4.5 Rational expression in \(e^{cx}\)

$$
\int \frac{ae^{cx}-1}{be^{cx}-1}\,dx=
\frac{(a-b)\ln(1-be^{cx})}{bc}+x+C
$$

This can be checked by differentiating the right-hand side directly.

### 4.6 Exponential-times-operator identity

$$
\int e^x\left(f(x)+f'(x)\right)\,dx=
e^x f(x)+C
$$

This is immediate from

\[
\frac{d}{dx}\left(e^x f(x)\right)=e^x(f+f')
\]

### 4.7 Higher-order identity with \(e^x\)

$$
\int e^x\left(
f(x)-(-1)^n\frac{d^n f(x)}{dx^n}
\right)\,dx=
e^x\sum_{k=1}^{n}(-1)^{k-1}\frac{d^{k-1}f(x)}{dx^{k-1}}+C
$$

This is a repeated integration-by-parts identity.

### 4.8 Higher-order identity with \(e^{-x}\)

$$
\int e^{-x}\left(
f(x)-\frac{d^n f(x)}{dx^n}
\right)\,dx=
-\,e^{-x}\sum_{k=1}^{n}\frac{d^{k-1}f(x)}{dx^{k-1}}+C
$$

---

## 5. Short remarks

1. Several formulas on the Wikipedia page suppress domain restrictions. I restored the most obvious ones.
2. Some integrals are not elementary and therefore naturally involve:
   - \(\operatorname{Ei}\) (exponential integral),
   - \(\operatorname{erf}\) / \(\operatorname{erfi}\),
   - \(\Gamma(s,z)\) (upper incomplete gamma function).
3. For classroom use, the most important entries are usually:
   - \(\int e^{cx}dx\),
   - \(\int a^x dx\),
   - \(\int x^n e^{cx}dx\),
   - Gaussian integrals involving \(\operatorname{erf}\),
   - the reduction formulas.

---

## 6. Minimal verification strategy

To verify any entry quickly:

- Differentiate the claimed antiderivative.
- For forms like \(\int x^n e^{cx}dx\), use integration by parts.
- For forms involving \(e^{g(x)}\), look for a substitution \(u=g(x)\).
- For non-elementary cases, recognize the standard special functions rather than forcing an elementary answer.

---

## 7. Caveat

This file is a cleaned study note based on the formulas listed on that Wikipedia page, not an independent scholarly source.  
For formal work, check a table of integrals or a CAS and verify the parameter assumptions.
