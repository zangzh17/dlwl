# Fresnel transform: 2 chirps

Fresnel transform:


$$
\begin{aligned}
U_{2}(X, Y)=&\frac{\exp (j k z)}{j \lambda z} \exp \left[\frac{j\pi}{\lambda z}\left(X^{2}+Y^{2}\right)\right] \\
& \times \iint\left\{U_{1}(x,y) \exp \left[\frac{j\pi}{\lambda z}\left(x^{2}+y^{2}\right)\right]\right\}\\ 
&\exp \left[-j2 \pi \left(x\frac{X}{\lambda z}+y\frac{Y}{\lambda z}\right)\right] d x d y .
\end{aligned}
$$

Inner chirp:

$$
\exp \left(\frac{j\pi}{\lambda z}(x^2+y^2)\right)
$$

Outer chirp:

$$
\exp \left(\frac{j\pi}{\lambda z}(X^2+Y^2)\right),\\
X = \lambda z f_x,\; Y = \lambda z f_y
$$

# decide N, pitch

In order to fulfill input field sampling requirement, we have

$$
L = \frac{\lambda z}{\Delta}-N\Delta 
= N\frac{\lambda z}{W}-W.
$$

Given the input field size $W$, output field size $L$, the sampling number is fixed, which is

$$
N = W\frac{L+W}{\lambda z}\approx 
\frac{L}{W}\frac{W^2}{\lambda z} = \frac{L}{W}N_F=mN_F,
$$
where m is magnification factor and $N_F$ is Fresnel number.

The sampling pitch is then

$$
\Delta = W/N = \frac{\lambda z}{L+W}.
$$

The output range of FFT is

$$
X/\lambda z\in [-\frac{1}{2}\frac{L+W}{\lambda z},\frac{1}{2}\frac{L+W}{\lambda z}]
$$

i.e.,

$$
X\in [-\frac{L+W}{2},\frac{L+W}{2}].
$$

As a result, the result should be cropped to be $\frac{L}{L+W}$ of the original FFT result.

The consequence of using larger $\Delta$:

- $L\leq\lambda z/\Delta$: aliasing error at edges
- $L>\lambda z/\Delta$: reduced output size $L$, with aliasing errors



# decide FFT points $\hat N$

In order to fulfill output field sampling requirement, we have

$$
\Delta^2 \geq \frac{\lambda z}{N+\hat{N}}.
$$

i.e.

$$
N\Delta^2+\hat N\Delta^2 \geq \lambda z.
$$

Combined with input field sampling requirement, we have
$$
\hat{N}\geq L/\Delta 
$$

i.e.

$$
\hat{N}/N\geq L/W=m.
$$

Output field points $M$ is

$$
M = \hat N \frac{L\Delta}{\lambda z} = \frac{\hat N}{N} \frac{L}{W} \frac{W^2}{\lambda z}= m \frac{\hat N}{N} N_F\geq m^2 N_F 
$$

The consequence of using $\hat N/N <m$:

- insufficient sampling of output field

