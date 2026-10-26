import numpy as np

A = np.asarray


class ProjectionMesh:
    def __init__(s, path):
        with np.load(path, allow_pickle=False) as d:
            g = lambda k, t, o=None: d[k].astype(t) if k in d else o
            s.name = str(d["name"])
            s.center_lon, s.center_lat = map(float, d["center_lonlat"])
            s.w, s.m, s.triangles, s.boundary_lonlat, s.boundary_m = (
                g("w", complex),
                g("m", complex),
                g("triangles", np.int64),
                g("boundary_lonlat", float),
                g("boundary_m", complex),
            )
            s.meters_per_unit = float(d["meters_per_unit"])
            s.poly_coef, s.poly_scale = g("poly_coef", complex), g("poly_scale", float)
            s.poly_on_sphere = bool(g("poly_on_sphere", bool, False))
            s.support_radius = g("support_radius", float)
            s.isoline_scale_factor, s.isoline_w, s.isoline_pointer = (
                g(
                    "isoline_scale_factor",
                    float,
                    (
                        d["isoline_area_factor"] ** 0.5
                        if "isoline_area_factor" in d
                        else None
                    ),
                ),
                g("isoline_w", complex),
                g("isoline_pointer", np.int64),
            )
            s.support_pointer, s.support_index = g("support_pointer", np.int64), g(
                "support_index", np.int64
            )
            s.minimum_scale_factor, s.minimum_lonlat = g(
                "minimum_scale_factor", float
            ), g("minimum_lonlat", float)
        s.path = path
        s._prepare()

    def _prepare(s):
        c = s.w[s.triangles]
        M = s.m[s.triangles]
        d = (
            lambda c: np.abs(
                np.linalg.det(np.stack([c, c.conj(), np.ones_like(c)], -1))
            )
            > 1e-300
        )
        S = np.stack([c, c.conj(), np.ones_like(c)], -1)
        s.usable = u = d(c)
        I = np.zeros_like(S)
        I[u] = np.linalg.inv(S[u])
        s.coef_a, s.coef_b, s.coef_c = ((I[:, k, :] * M).sum(1) for k in range(3))
        s.median_edge = float(np.median(np.abs(c - np.roll(c, 1, 1))))
        s._lw = Locator(c, u)
        s._lm = None
        s._um = u & d(M)

    def locate(s, wx, wy):
        return s._lw.locate(wx, wy)

    def nearest_triangle(s, wx, wy):
        return s._lw.nearest(wx, wy)

    def lonlat_to_w(s, lon, lat):
        l, p = np.radians(A(lon, float)), np.radians(A(lat, float))
        L, P = np.radians(s.center_lon), np.radians(s.center_lat)
        d = l - L
        x, y = np.cos(p) * np.sin(d), np.cos(P) * np.sin(p) - np.sin(P) * np.cos(
            p
        ) * np.cos(d)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 1 / (1 + np.sin(P) * np.sin(p) + np.cos(P) * np.cos(p) * np.cos(d))
            x, y = x * k, y * k
        return tuple(np.clip(np.nan_to_num(v), -1e6, 1e6) for v in (x, y))

    def w_to_m(s, wx, wy):
        m, o, _ = s._wm(wx, wy)
        return m.real, m.imag, o

    def _wm(s, wx, wy):
        wx, wy = A(wx, float).ravel(), A(wy, float).ravel()
        t = s.locate(wx, wy)
        o = t < 0
        if o.any():
            t[o] = s.nearest_triangle(wx[o], wy[o])
        q = wx + 1j * wy
        m = (
            s.coef_a[t] * q + s.coef_b[t] * q.conj() + s.coef_c[t]
            if s.poly_coef is None
            else s._blend(t, q) if s.support_radius is None else s._smooth(t, q)
        )
        return m, o, t

    def m_to_w(s, mx, my, it=8, tol=0.01):
        if s._lm is None:
            s._lm = Locator(s.m[s.triangles], s._um)
        mx, my = A(mx, float).ravel(), A(my, float).ravel()
        t = s._lm.locate(mx, my)
        o = t < 0
        if o.any():
            t[o] = s._lm.nearest(mx[o], my[o])
        q = mx + 1j * my
        a, b = s.coef_a[t], s.coef_b[t]
        d = q - s.coef_c[t]
        w = (a.conj() * d - b * d.conj()) / (abs(a) ** 2 - abs(b) ** 2)
        k = np.arange(q.size)
        for _ in range(it * (s.poly_coef is not None)):
            f, _, t = s._wm(w[k].real, w[k].imag)
            e = q[k] - f
            w[k] += e / s.coef_a[t]
            k = k[np.abs(e) * s.meters_per_unit > tol]
            if not k.size:
                break
        return w.real, w.imag, o

    def w_to_lonlat(s, x, y):
        x, y = A(x, float), A(y, float)
        r = np.hypot(x, y)
        c = 2 * np.arctan(r)
        P = np.radians(s.center_lat)
        a, b = np.sin(c), np.cos(c)
        la = np.arcsin(
            np.clip(
                b * np.sin(P)
                + np.divide(y * a * np.cos(P), r, out=np.zeros_like(r), where=r > 0),
                -1,
                1,
            )
        )
        lo = np.radians(s.center_lon) + np.arctan2(
            x * a, r * np.cos(P) * b - y * np.sin(P) * a
        )
        return (np.degrees(lo) + 180) % 360 - 180, np.degrees(la)

    def _fit(s, v, q):
        c = s.w[v]
        l = (q - c) / (1 + c.conj() * q) if s.poly_on_sphere else q - c
        z = l / s.poly_scale[v]
        k = s.poly_coef[v]
        r = k[..., -1]
        for j in range(k.shape[-1] - 2, -1, -1):
            r = r * z + k[..., j]
        return r, l

    def _smooth(s, t, q):
        r = np.empty(q.shape, complex)
        for j in range(0, q.size, 20000):
            T, Q = t[j : j + 20000], q[j : j + 20000]
            a = s.support_pointer[T]
            n = s.support_pointer[T + 1] - a
            p = np.repeat(np.arange(T.size), n)
            v = s.support_index[
                np.repeat(a, n) + np.arange(p.size) - np.repeat(np.cumsum(n) - n, n)
            ]
            f, l = s._fit(v, Q[p])
            R = s.support_radius[v]
            w = (
                np.clip(
                    1
                    - np.divide(np.abs(l), R, out=np.full(R.shape, np.inf), where=R > 0)
                    ** 2,
                    0,
                    None,
                )
                ** 3
            )
            W = np.bincount(p, w, T.size)
            r[j : j + 20000] = (
                np.bincount(p, w * f.real, T.size)
                + 1j * np.bincount(p, w * f.imag, T.size)
            ) / np.where(W > 0, W, 1)
            b = np.where(W == 0)[0]
            if b.size:
                r[j + b] = s._blend(T[b], Q[b])
        return r

    def _blend(s, t, q):
        v = s.triangles[t]
        c = s.w[v]
        e, f = c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]
        a = (e * f.conj()).imag
        b1, b2 = ((q - c[:, 0]) * f.conj()).imag / a, (
            e * (q - c[:, 0]).conj()
        ).imag / a
        return (np.stack([1 - b1 - b2, b1, b2], 1) * s._fit(v, q[:, None])[0]).sum(1)

    def log_scale(s, q, h=1e-6):
        f = lambda z: (lambda a, b, _: a + 1j * b)(*s.w_to_m(z.real, z.imag))
        q = A(q, complex).ravel()
        return np.log(
            np.abs(f(q + h) - f(q - h))
            / (2 * h)
            * (1 + np.abs(q) ** 2)
            / 2
            * s.meters_per_unit
            / 6378137
        )

    def nodal_log_scale(s):
        if getattr(s, "_nodal", None) is None:
            u = np.unique(s.triangles[s.usable])
            s._nodal = np.full(len(s.w), np.nan)
            s._nodal[u] = s.log_scale(s.w[u])
        return s._nodal

    def isolines(s, level, iterations=3):
        v = s.nodal_log_scale()[s.triangles[s.usable]]
        T = s.triangles[s.usable]
        b = v > level
        x = np.where((b.sum(1) % 3) != 0)[0]
        E = {}
        G = {}
        for t in x:
            k = []
            for i, j in ((0, 1), (1, 2), (2, 0)):
                if b[t, i] != b[t, j]:
                    e = (min(T[t, i], T[t, j]), max(T[t, i], T[t, j]))
                    if e not in E:
                        a, c = (i, j) if T[t, i] < T[t, j] else (j, i)
                        r = (level - v[t, a]) / (v[t, c] - v[t, a])
                        E[e] = s.w[T[t, a]] + r * (s.w[T[t, c]] - s.w[T[t, a]])
                    k.append(e)
            if len(k) == 2:
                G.setdefault(k[0], []).append(k[1])
                G.setdefault(k[1], []).append(k[0])
        L = []
        seen = set()
        for e0 in sorted(G, key=lambda e: len(G[e])):
            if e0 in seen:
                continue
            c = [e0]
            seen.add(e0)
            for d in (0, 1):
                while True:
                    n = [f for f in G[c[-1]] if f not in seen]
                    if not n:
                        break
                    c.append(n[0])
                    seen.add(n[0])
                if d == 0:
                    c.reverse()
            if len(c) > 2 and c[0] in G[c[-1]]:
                c.append(c[0])
            L.append(np.array([E[e] for e in c]))
        h = s.median_edge / 4
        o = []
        for l in L:
            if l.size < 2:
                continue
            n = np.maximum(1, np.ceil(np.abs(np.diff(l)) / h).astype(int))
            q = np.concatenate(
                [
                    np.repeat(l[:-1], n)
                    + np.concatenate([np.arange(k) / k for k in n])
                    * np.repeat(np.diff(l), n),
                    l[-1:],
                ]
            )
            for _ in range(iterations):
                e = 1e-5 * h
                F = s.log_scale(q)
                g = (
                    s.log_scale(q + e)
                    - s.log_scale(q - e)
                    + 1j * (s.log_scale(q + 1j * e) - s.log_scale(q - 1j * e))
                ) / (2 * e)
                d = -(F - level) * g / np.maximum(np.abs(g) ** 2, 1e-300)
                q = q + d * np.minimum(1, h / np.maximum(np.abs(d), 1e-300))
            o.append(q)
        return o

    def boundary_w(s):
        return s.lonlat_to_w(s.boundary_lonlat[:, 0], s.boundary_lonlat[:, 1])

    def boundary_mapped(s, step_m=1000):
        x, y = s.boundary_w()
        w = x + 1j * y
        n = np.maximum(
            1,
            np.ceil(
                np.abs(np.diff(s.boundary_m)) / (step_m / s.meters_per_unit)
            ).astype(int),
        )
        d = np.concatenate(
            [
                np.repeat(w[:-1], n)
                + np.concatenate([np.arange(k) / k for k in n])
                * np.repeat(np.diff(w), n),
                w[-1:],
            ]
        )
        x, y, _ = s.w_to_m(d.real, d.imag)
        return x + 1j * y


class Locator:
    def __init__(s, c, u):
        s.tri_x, s.tri_y, s.centroids, s.usable_idx = (
            c.real,
            c.imag,
            c.mean(1),
            np.where(u)[0],
        )
        n = max(8, int(len(c) ** 0.5))
        s.x0, s.y0 = x0, y0 = float(c.real.min()), float(c.imag.min())
        s.side = n
        s.cell = h = max(c.real.max() - x0, c.imag.max() - y0) / n * (1 + 1e-9)
        f = lambda v, o: np.clip(((v - o) / h).astype(np.int64), 0, n - 1)
        X, Y = c.real[u], c.imag[u]
        a, b, e, k = f(X.min(1), x0), f(X.max(1), x0), f(Y.min(1), y0), f(Y.max(1), y0)
        w = b - a + 1
        N = w * (k - e + 1)
        j = np.arange(N.sum()) - np.repeat(np.cumsum(N) - N, N)
        W = np.repeat(w, N)
        g = (np.repeat(e, N) + j // W) * n + np.repeat(a, N) + j % W
        o = np.argsort(g, kind="stable")
        s.index = np.repeat(s.usable_idx, N)[o]
        s.pointer = np.concatenate([[0], np.cumsum(np.bincount(g, minlength=n * n))])

    def locate(s, wx, wy):
        n = s.side
        gx, gy = np.floor((wx - s.x0) / s.cell), np.floor((wy - s.y0) / s.cell)
        r = np.full(wx.shape, -1, np.int64)
        i = np.where((gx >= 0) & (gx < n) & (gy >= 0) & (gy < n))[0]
        g = (gy[i] * n + gx[i]).astype(np.int64)
        a = s.pointer[g]
        N = s.pointer[g + 1] - a
        for j0 in range(0, i.size, 100000):
            j1 = min(i.size, j0 + 100000)
            m = N[j0:j1]
            p = np.repeat(np.arange(j0, j1), m)
            t = s.index[
                np.repeat(a[j0:j1], m)
                + np.arange(p.size)
                - np.repeat(np.cumsum(m) - m, m)
            ]
            px, py = wx[i[p]], wy[i[p]]
            X, Y = [s.tri_x[t, k] for k in (0, 1, 2, 0)], [
                s.tri_y[t, k] for k in (0, 1, 2, 0)
            ]
            D = [
                (X[k + 1] - X[k]) * (py - Y[k]) - (Y[k + 1] - Y[k]) * (px - X[k])
                for k in range(3)
            ]
            h = ((D[0] >= -1e-12) & (D[1] >= -1e-12) & (D[2] >= -1e-12)) | (
                (D[0] <= 1e-12) & (D[1] <= 1e-12) & (D[2] <= 1e-12)
            )
            r[i[p[h]]] = t[h]
        return r

    def nearest(s, wx, wy):
        c = s.centroids[s.usable_idx]
        r = np.empty(wx.shape, np.int64)
        k = max(1, 4000000 // max(c.size, 1))
        for j in range(0, wx.size, k):
            r[j : j + k] = s.usable_idx[
                np.abs((wx[j : j + k] + 1j * wy[j : j + k])[:, None] - c[None]).argmin(
                    1
                )
            ]
        return r