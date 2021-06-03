1. $X_{0}=X ; y_{0}=y$
2. Pour $k=1,2,...,s$ :
    1. $w_{k}=\frac{X_{k-1}^{T} y_{k-1}}{y_{k-1}^{T} y_{k-1}}$
    2. Normer $w_k$ à 1
    3. $t_{k}=\frac{X_{k-1} w_{k}}{w_{k}^{T} w_{k}}$
    4. $p_{k}=\frac{X_{k-1}^{T} t_{k}}{t_{k}^{T} t_{k}}$
    5. $X_{k}=X_{k-1}-t_{k} p_{k}^{T}$
    6. $q_{k}=\frac{y_{k-1}^{T} t_{k}}{t_{k}^{T} t_{k}}$
    7. $u_{k}=\frac{y_{k-1}}{q_{k}}$
    8. $y_{k}=y_{k-1}-q_{k} t_{k}$

**Commentaires** Lorsqu'il n'y a pas de données manquantes, on peut remplacer les étapes $2.1$ et $2.2$ par $w_{k}=\frac{X_{k-1}^{T} y_{k-1}}{\left\|X_{k-1}^{T} y_{k-1}\right\|}$ et l'étape $2.3$ par $t_{k}=X_{k-1}w_{k}$

Pour avoir $W$ tel que $T=XW$ on calcule :
$$\mathbf{W}=\mathbf{W}^{*}\left(\widetilde{\mathbf{P}} \mathbf{W}^{*}\right)^{-1}$$
où $\widetilde{\mathbf{P}}_{K \times p}=\mathbf{t}\left[p_{1}, \ldots, p_{K}\right]$ et $\mathbf{W}^{*}_{p \times K} = [w_1, \ldots, w_K]$