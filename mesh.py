import numpy as np

A = np.asarray


class ProjectionMesh:
    def __init__(s, path):
        with np.load(path, allow_pickle=False) as d:
            if int(d["format_version"]) > 2:
                raise ValueError(
                    f'{path} uses projection format {int(d["format_version"])}, this plugin reads up to 2'
                )
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
        S = np.stack([c, c.conj(), np.ones_like(c)], -1)
        s.usable = u = np.abs(np.linalg.det(S)) > 1e-300
        I = np.zeros_like(S)
        I[u] = np.linalg.inv(S[u])
        M = s.m[s.triangles]
        s.coef_a, s.coef_b, s.coef_c = ((I[:, k, :] * M).sum(1) for k in range(3))
        s.tri_x, s.tri_y, s.centroids, s._usable_idx = (
            c.real,
            c.imag,
            c.mean(1),
            np.where(u)[0],
        )
        s.median_edge = float(np.median(np.abs(c - np.roll(c, 1, 1))))
        n = max(8, int(len(s.triangles) ** 0.5))
        s.grid_x0, s.grid_y0 = x0, y0 = float(c.real.min()), float(c.imag.min())
        s.grid_side = n
        s.grid_cell = h = max(c.real.max() - x0, c.imag.max() - y0) / n * (1 + 1e-9)
        f = lambda v, o: np.clip(((v - o) / h).astype(np.int64), 0, n - 1)
        X, Y = c.real, c.imag
        a, b, e, k = f(X.min(1), x0), f(X.max(1), x0), f(Y.min(1), y0), f(Y.max(1), y0)
        B = [[] for _ in range(n * n)]
        for t in s._usable_idx:
            for gy in range(e[t], k[t] + 1):
                for gx in range(a[t], b[t] + 1):
                    B[gy * n + gx].append(t)
        s.grid = np.full((n * n, max(1, max(map(len, B)))), -1, np.int64)
        for i, l in enumerate(B):
            s.grid[i, : len(l)] = l

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

    def locate(s, wx, wy):
        r = np.full(wx.shape, -1, np.int64)
        k = max(1, 1000000 // s.grid.shape[1])
        for j in range(0, wx.size, k):
            r[j : j + k] = s._locate(wx[j : j + k], wy[j : j + k])
        return r

    def _locate(s, wx, wy):
        n = s.grid_side
        gx, gy = np.floor((wx - s.grid_x0) / s.grid_cell), np.floor(
            (wy - s.grid_y0) / s.grid_cell
        )
        r = np.full(wx.shape, -1, np.int64)
        i = np.where((gx >= 0) & (gx < n) & (gy >= 0) & (gy < n))[0]
        if i.size == 0:
            return r
        C = s.grid[(gy[i] * n + gx[i]).astype(np.int64)]
        t = np.maximum(C, 0)
        px, py = wx[i, None], wy[i, None]
        X, Y = [s.tri_x[t, j] for j in (0, 1, 2, 0)], [
            s.tri_y[t, j] for j in (0, 1, 2, 0)
        ]
        D = [
            (X[j + 1] - X[j]) * (py - Y[j]) - (Y[j + 1] - Y[j]) * (px - X[j])
            for j in range(3)
        ]
        o = (C >= 0) & (
            ((D[0] >= -1e-12) & (D[1] >= -1e-12) & (D[2] >= -1e-12))
            | ((D[0] <= 1e-12) & (D[1] <= 1e-12) & (D[2] <= 1e-12))
        )
        h = o.any(1)
        r[i[h]] = C[h, o.argmax(1)[h]]
        return r

    def nearest_triangle(s, wx, wy):
        c = s.centroids[s._usable_idx]
        r = np.empty(wx.shape, np.int64)
        k = max(1, 4000000 // max(c.size, 1))
        for j in range(0, wx.size, k):
            r[j : j + k] = s._usable_idx[
                np.abs((wx[j : j + k] + 1j * wy[j : j + k])[:, None] - c[None]).argmin(
                    1
                )
            ]
        return r

    def w_to_m(s, wx, wy):
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
        return m.real, m.imag, o

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
