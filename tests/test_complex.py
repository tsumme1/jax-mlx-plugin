from jax import random, grad, jvp

def check(seed):
  key = random.key(seed)

  # random coeffs for u and v
  key, subkey = random.split(key)
  a, b, c, d = random.uniform(subkey, (4,))

  def fun(z):
    x, y = jnp.real(z), jnp.imag(z)
    return u(x, y) + v(x, y) * 1j

  def u(x, y):
    return a * x + b * y

  def v(x, y):
    return c * x + d * y

  # primal point
  key, subkey = random.split(key)
  x, y = random.uniform(subkey, (2,))
  z = x + y * 1j

  # tangent vector
  key, subkey = random.split(key)
  c, d = random.uniform(subkey, (2,))
  z_dot = c + d * 1j

  # check jvp
  _, ans = jvp(fun, (z,), (z_dot,))
  expected = (grad(u, 0)(x, y) * c +
              grad(u, 1)(x, y) * d +
              grad(v, 0)(x, y) * c * 1j+
              grad(v, 1)(x, y) * d * 1j)
  print(jnp.allclose(ans, expected))