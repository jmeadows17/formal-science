## Example 1

### Q1

Consider a heterogeneous tumour composed of a drug-sensitive subpopulation and a drug-resistant subpopulation competing through Lotka--Volterra interactions under continuous adaptive chemotherapy. Let $t\in[0,\infty)$ denote time, let $x(t)>0$ be the normalized abundance of drug-sensitive tumour cells, let $y(t)\geq 0$ be the normalized abundance of drug-resistant tumour cells, and let $T(t)=x(t)+y(t)$ be the total tumour burden. Let $r_x>0$ and $r_y>0$ be the intrinsic growth rates of the sensitive and resistant subpopulations, let $\alpha\geq 0$ be the competitive effect of resistant cells on sensitive cells, let $\beta\geq 0$ be the competitive effect of sensitive cells on resistant cells, and let $u(t)\geq 0$ be the continuous per-capita drug-induced removal rate acting only on sensitive cells. Assume $x(t)$ and $y(t)$ are differentiable, and assume chemotherapy directly removes sensitive cells but has no direct killing effect on resistant cells. The governing equations are

$$
\frac{dx}{dt}=r_xx(t)\left(1-x(t)-\alpha y(t)\right)-u(t)x(t)
$$

$$
\frac{dy}{dt}=r_yy(t)\left(1-y(t)-\beta x(t)\right)
$$

Assume continuous adaptive therapy holds the total tumour burden fixed at a control value $A>0$, so that

$$
T(t)=x(t)+y(t)=A
$$

and

$$
\frac{dT}{dt}=0
$$

Show that

$$
u(t)=
\frac{
r_xx(t)\left(1-x(t)-\alpha y(t)\right)
+
r_yy(t)\left(1-y(t)-\beta x(t)\right)
}{
x(t)
}
$$

### A1

From the question, $T(t)=x(t)+y(t)$, so by differentiation of a sum,

$$
\frac{dT}{dt}=\frac{dx}{dt}+\frac{dy}{dt}
$$

From the question, continuous adaptive therapy holds the total tumour burden fixed, so

$$
0=\frac{dx}{dt}+\frac{dy}{dt}
$$

Because chemotherapy directly removes sensitive cells but has no direct killing effect on resistant cells, the total-burden control balance contains one drug-removal flux, $u(t)x(t)$, and that flux appears only in the sensitive-cell equation.

From the question, the sensitive-cell and resistant-cell equations hold, so substitution gives

$$
0=
r_xx(t)\left(1-x(t)-\alpha y(t)\right)-u(t)x(t)
+
r_yy(t)\left(1-y(t)-\beta x(t)\right)
$$

By algebraic rearrangement,

$$
u(t)x(t)=
r_xx(t)\left(1-x(t)-\alpha y(t)\right)
+
r_yy(t)\left(1-y(t)-\beta x(t)\right)
$$

From the question, $x(t)>0$, so division by $x(t)$ gives

$$
u(t)=
\frac{
r_xx(t)\left(1-x(t)-\alpha y(t)\right)
+
r_yy(t)\left(1-y(t)-\beta x(t)\right)
}{
x(t)
}
$$

Biologically, this continuous dose removes enough sensitive-cell mass to offset the net growth of the whole tumour at the chosen burden. Resistant cells are not killed directly; they are controlled indirectly by maintaining competition within the tumour.

---

## Example 2

### Q2

Consider a population divided into finitely many biological types whose abundances change through differential growth. Let $m\in\mathbb{N}$, let $J=\{1,\ldots,m\}$ be the finite set of types, let $n_j(t)>0$ be the abundance of type $j\in J$, let $N(t)=\sum_{j\in J}n_j(t)$ be the total population size, and let $p_j(t)=n_j(t)/N(t)$ be the frequency of type $j$. Let $r_j(t)=\dot n_j(t)/n_j(t)$ be the growth rate of type $j$, let $a_j(t)\in\mathbb{R}$ be the value of a quantitative trait $A$ in type $j$, and let $\dot A=(\dot a_j(t))_{j\in J}$ with $\dot a_j(t)=da_j(t)/dt$. For any type-indexed quantity $B=(b_j(t))_{j\in J}$, define the population average by

$$
\langle B\rangle=\sum_{j\in J}p_j(t)b_j(t)
$$

Define $A^2=(a_j(t)^2)_{j\in J}$, define $r^2=(r_j(t)^2)_{j\in J}$, define $Ar=(a_j(t)r_j(t))_{j\in J}$, and define the covariance by

$$
\operatorname{cov}(A,r)=\langle Ar\rangle-\langle A\rangle\langle r\rangle
$$

Let $\sigma_A\geq 0$ and $\sigma_r\geq 0$ be the standard deviations defined by

$$
\sigma_A^2=\langle A^2\rangle-\langle A\rangle^2
$$

$$
\sigma_r^2=\langle r^2\rangle-\langle r\rangle^2
$$

Assume all quantities are differentiable and that the governing abundance equation is

$$
\dot n_j(t)=r_j(t)n_j(t)
$$

for every $j\in J$. Show that

$$
\left|\frac{d\langle A\rangle}{dt}-\langle \dot A\rangle\right|\leq \sigma_A\sigma_r
$$

### A2

From the question, $\langle A\rangle=\sum_{j\in J}p_j(t)a_j(t)$, so by the product rule,

$$
\frac{d\langle A\rangle}{dt}
=
\sum_{j\in J}\dot p_j(t)a_j(t)
+
\langle \dot A\rangle
$$

By algebraic subtraction,

$$
\frac{d\langle A\rangle}{dt}-\langle \dot A\rangle
=
\sum_{j\in J}\dot p_j(t)a_j(t)
$$

Because biological type frequencies change through differential growth of type abundances, the reweighting term is determined by the frequency derivative induced by the abundance equation.

From the question, $p_j(t)=n_j(t)/N(t)$ and $\dot n_j(t)=r_j(t)n_j(t)$, so by the quotient rule and summation over types,

$$
\dot p_j(t)=p_j(t)\left(r_j(t)-\langle r\rangle\right)
$$

By substitution,

$$
\frac{d\langle A\rangle}{dt}-\langle \dot A\rangle
=
\sum_{j\in J}p_j(t)a_j(t)\left(r_j(t)-\langle r\rangle\right)
$$

By expansion of the right-hand side,

$$
\frac{d\langle A\rangle}{dt}-\langle \dot A\rangle
=
\langle Ar\rangle-\langle A\rangle\langle r\rangle
$$

From the question, $\operatorname{cov}(A,r)=\langle Ar\rangle-\langle A\rangle\langle r\rangle$, so by substitution and the Cauchy--Schwarz inequality for covariance,

$$
\left|\frac{d\langle A\rangle}{dt}-\langle \dot A\rangle\right|\leq \sigma_A\sigma_r
$$

Biologically, the mean trait can change rapidly through population reweighting only when there is variation in the trait and variation in growth rate. If all types have nearly the same trait value or nearly the same growth rate, differential growth cannot strongly shift the trait mean.

---

## Example 3

### Q3

Consider a quantitative biological trait in an outcrossing population whose phenotype is the sum of an inherited genetic component and an environmental component. Let $G_m\in\mathbb{R}$ and $G_f\in\mathbb{R}$ be the genetic components of the two parents, let $G_o\in\mathbb{R}$ be the genetic component of one offspring, let $E_o\in\mathbb{R}$ be the offspring environmental deviation, and let $P_o\in\mathbb{R}$ be the offspring phenotype. Let $g_m,g_f\in\mathbb{R}$ be fixed parental genetic values, let $V_S>0$ be the segregation variance, and let $V_E>0$ be the environmental variance. Let $\Delta\in\mathbb{R}$ be the Mendelian segregation residual, and let $\mathcal{N}(m,v)$ denote a normal distribution with mean $m$ and variance $v$. Assume infinitesimal inheritance, so that conditional on $G_m=g_m$ and $G_f=g_f$,

$$
\Delta\sim\mathcal{N}(0,V_S)
$$

and $\Delta$ is independent of the parental genetic values. Assume environmental noise satisfies

$$
E_o\sim\mathcal{N}(0,V_E)
$$

and $E_o$ is independent of $\Delta$ and of the parental genetic values. The governing inheritance model is

$$
G_o=\frac{G_m+G_f}{2}+\Delta
$$

The governing phenotype model is

$$
P_o=G_o+E_o
$$

Show that

$$
P_o\mid(G_m=g_m,G_f=g_f)\sim\mathcal{N}\left(\frac{g_m+g_f}{2},V_S+V_E\right)
$$

### A3

From the question, conditional on $G_m=g_m$ and $G_f=g_f$, the inheritance model gives

$$
G_o=\frac{g_m+g_f}{2}+\Delta
$$

Because infinitesimal inheritance treats Mendelian segregation as a parental-value-independent normal residual around the mid-parent genetic value, we can write

$$
G_o\mid(G_m=g_m,G_f=g_f)\sim\mathcal{N}\left(\frac{g_m+g_f}{2},V_S\right)
$$

From the question, $P_o=G_o+E_o$, so by substitution,

$$
P_o=\frac{g_m+g_f}{2}+\Delta+E_o
$$

Because the biological phenotype is the additive combination of inherited genetic value and independent environmental deviation, the within-family phenotypic deviation from the mid-parent genetic value is the sum of the segregation residual and the environmental deviation.

From the question, $\Delta\sim\mathcal{N}(0,V_S)$ and $E_o\sim\mathcal{N}(0,V_E)$ are independent, so by addition of independent normal random variables,

$$
\Delta+E_o\sim\mathcal{N}(0,V_S+V_E)
$$

By translation of this normal random variable,

$$
P_o\mid(G_m=g_m,G_f=g_f)\sim\mathcal{N}\left(\frac{g_m+g_f}{2},V_S+V_E\right)
$$

Biologically, offspring phenotypes are centred on the mid-parent genetic value, while within-family variation comes from Mendelian segregation and environmental noise. The key infinitesimal-model point is that this within-family variance does not depend on whether the parents have high or low genetic values.

---

## Example 4

### Q4

Consider a competitive ecological community of $n\in\mathbb{N}$ species governed by generalized Lotka--Volterra dynamics. Let $x(t)=(x_1(t),\ldots,x_n(t))^\top\in(0,\infty)^n$ be the vector of species densities, let $r_i>0$ be the intrinsic growth rate of species $i$, let $\mathbf{1}\in\mathbb{R}^n$ be the vector with every component equal to $1$, and let $I\in\mathbb{R}^{n\times n}$ be the identity matrix. Let $B\in\mathbb{R}^{n\times n}$ be an interspecific competition matrix with $B_{ii}=0$ and $B_{ij}\geq 0$ for $i\neq j$. Let $\lVert\cdot\rVert_\infty$ denote the maximum absolute row-sum matrix norm for matrices and the maximum absolute component norm for vectors, and define

$$
\beta=\lVert B\rVert_\infty=\max_{1\leq i\leq n}\sum_{j=1}^n |B_{ij}|
$$

Let $\alpha>0$ be the strength of intraspecific competition, assume

$$
\alpha>2\beta
$$

and define

$$
A_\alpha=\alpha I+B
$$

For vectors, let $v>\mathbf{0}$ mean that every component of $v$ is strictly positive. The governing equation is

$$
\frac{dx_i}{dt}
=
x_i(t)r_i\left(1-\sum_{j=1}^n(A_\alpha)_{ij}x_j(t)\right)
$$

for $i=1,\ldots,n$. Show that a feasible coexistence equilibrium exists and is given by

$$
x^*(\alpha)=A_\alpha^{-1}\mathbf{1}>\mathbf{0}
$$

### A4

Because feasible ecological coexistence requires all species densities to be positive and all species per-capita growth rates to vanish, it is enough to construct a vector $x^*(\alpha)>\mathbf{0}$ satisfying $A_\alpha x^*(\alpha)=\mathbf{1}$.

From the question, $\beta=\lVert B\rVert_\infty$ and $\alpha>2\beta$, so by scalar division and the Neumann-series invertibility criterion,

$$
A_\alpha^{-1}
=
\frac{1}{\alpha}\left(I+\frac{B}{\alpha}\right)^{-1}
$$

By definition, set

$$
x^*(\alpha)=A_\alpha^{-1}\mathbf{1}
$$

By algebraic substitution and the inverse-difference identity,

$$
x^*(\alpha)-\frac{1}{\alpha}\mathbf{1}
=
-\frac{1}{\alpha}
\left(I+\frac{B}{\alpha}\right)^{-1}
\frac{B}{\alpha}\mathbf{1}
$$

By the matrix norm bound,

$$
\left\lVert x^*(\alpha)-\frac{1}{\alpha}\mathbf{1}\right\rVert_\infty
\leq
\frac{\beta}{\alpha(\alpha-\beta)}
$$

By componentwise comparison and the condition $\alpha>2\beta$, for every species $i$,

$$
x_i^*(\alpha)
\geq
\frac{\alpha-2\beta}{\alpha(\alpha-\beta)}
>
0
$$

From the definition of $x^*(\alpha)$, matrix multiplication gives

$$
A_\alpha x^*(\alpha)=\mathbf{1}
$$

By substitution into the generalized Lotka--Volterra equation,

$$
\frac{dx_i}{dt}
=
x_i^*(\alpha)r_i(1-1)
=
0
$$

By combining positivity with the equilibrium identity, the feasible coexistence equilibrium is

$$
x^*(\alpha)=A_\alpha^{-1}\mathbf{1}>\mathbf{0}
$$

Biologically, sufficiently strong intraspecific competition makes each species self-limiting enough that interspecific competitive effects cannot destroy positivity of the coexistence equilibrium.

---

## Example 5

### Q5

Consider a biochemical reaction network whose steady states are constrained by an algebraic invariant encoding robust perfect adaptation. Let $n,q\in\mathbb{N}$, let $x(t)=(x_1(t),\ldots,x_n(t))\in(0,\infty)^n$ be the concentration vector of $n$ molecular species, let $U\subseteq\mathbb{R}^q$ be a disturbance-parameter set, let $u\in U$ be a constant disturbance parameter, and let $f_i:(0,\infty)^n\times U\to\mathbb{R}$ be the rate equation for species $i$. Let $k\in\{1,\ldots,n\}$ index the molecular output species, let $c>0$ be its setpoint, let $h_i:(0,\infty)^n\times U\to\mathbb{R}$ be auxiliary algebraic functions, and let $g:(0,\infty)^n\times U\to\mathbb{R}$ be a gain function. The governing differential equations are

$$
\frac{dx_i}{dt}=f_i(x(t),u)
$$

for $i=1,\ldots,n$. Assume the network admits the adaptation-conferring algebraic invariant

$$
\sum_{i=1}^n h_i(x,u)f_i(x,u)
=
g(x,u)(x_k-c)
$$

for all positive concentration vectors $x$ and all disturbances $u\in U$. Assume $x^*(u)\in(0,\infty)^n$ is a positive steady state, with $k$th component $x_k^*(u)$, so that

$$
f_i(x^*(u),u)=0
$$

for every $i=1,\ldots,n$. Assume also that

$$
g(x^*(u),u)\neq 0
$$

Show that the output concentration at steady state is fixed at the setpoint,

$$
x_k^*(u)=c
$$

### A5

From the question, $x^*(u)$ is a positive steady state, so for every $i=1,\ldots,n$,

$$
f_i(x^*(u),u)=0
$$

By multiplication by $h_i(x^*(u),u)$ and summation over $i$,

$$
\sum_{i=1}^n h_i(x^*(u),u)f_i(x^*(u),u)=0
$$

Because robust perfect adaptation is a steady-state property of a biochemical network under constant disturbance, we evaluate the adaptation-conferring algebraic invariant at the positive steady state $x^*(u)$.

From the question, the algebraic invariant holds for all positive concentration vectors, so evaluation at $x=x^*(u)$ gives

$$
\sum_{i=1}^n h_i(x^*(u),u)f_i(x^*(u),u)
=
g(x^*(u),u)(x_k^*(u)-c)
$$

By substitution of the zero steady-state sum,

$$
0=g(x^*(u),u)(x_k^*(u)-c)
$$

From the question, $g(x^*(u),u)\neq 0$, so division by $g(x^*(u),u)$ gives

$$
x_k^*(u)-c=0
$$

By algebraic rearrangement,

$$
x_k^*(u)=c
$$

Biologically, the invariant acts as an embedded adaptation constraint: whenever the reaction network reaches a positive steady state and the gain factor does not vanish, the output species is forced to its molecular setpoint independently of the constant disturbance.

---
