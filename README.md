# algebra-lineal-nump
Resolución de sistemas de ecuaciones lineales con Python y NumPy
Codigo 1
import numpy as np

# Matriz de coeficientes
A = np.array([
    [2, 1, 1],
    [1, 3, 2],
    [3, 2, 3]
], dtype=float)

# Vector de resultados
B = np.array([1, 12, 13], dtype=float)

# Resolver sistema
sol = np.linalg.solve(A, B)

# Mostrar resultados
x, y, z = sol

print("Solución del sistema:")
print("x =", x)
print("y =", y)
print("z =", z)

# Verificación rápida
print("\nVerificación (Ax ≈ B):")
print(np.dot(A, sol))

Codigo 2
# ============================================================
#   RESOLUCIÓN DE SISTEMAS DE ECUACIONES LINEALES CON NumPy
#   Álgebra Lineal | Ciclo 1A | Google Colab
# ============================================================
#
#   Sistema a resolver:
#       2x +  y +  z =  1
#        x + 3y + 2z = 12
#       3x + 2y + 3z = 13
# ============================================================

import numpy as np

# ── Separador visual ─────────────────────────────────────────
def separador(titulo=""):
    print("\n" + "═" * 55)
    if titulo:
        print(f"  {titulo}")
        print("═" * 55)

# ════════════════════════════════════════════════════════════
# PASO 1: Presentación del sistema
# ════════════════════════════════════════════════════════════
separador("SISTEMA DE ECUACIONES LINEALES")
print("""
   Ecuación 1:   2x +  y +  z =  1
   Ecuación 2:    x + 3y + 2z = 12
   Ecuación 3:   3x + 2y + 3z = 13
""")

# ════════════════════════════════════════════════════════════
# PASO 2: Definir la matriz de coeficientes A (3×3)
#
#   Cada fila representa los coeficientes de una ecuación.
#   Cada columna corresponde a una incógnita: x, y, z.
#
#       A = | 2  1  1 |
#           | 1  3  2 |
#           | 3  2  3 |
# ════════════════════════════════════════════════════════════
A = np.array([
    [2, 1, 1],   # coeficientes de la ecuación 1
    [1, 3, 2],   # coeficientes de la ecuación 2
    [3, 2, 3]    # coeficientes de la ecuación 3
], dtype=float)

separador("PASO 1 · Matriz de coeficientes A")
print(f"\n   A =\n")
for fila in A:
    print("      ", fila)

# ════════════════════════════════════════════════════════════
# PASO 3: Definir el vector de resultados B
#
#   Contiene los términos independientes (lado derecho).
#       B = [1, 12, 13]
# ════════════════════════════════════════════════════════════
B = np.array([1, 12, 13], dtype=float)

separador("PASO 2 · Vector de resultados B")
print(f"\n   B = {B}\n")

# ════════════════════════════════════════════════════════════
# PASO 4: Verificar que el sistema tiene solución única
#
#   El sistema Ax = B tiene solución única si y solo si
#   det(A) ≠ 0 (la matriz A es invertible / no singular).
# ════════════════════════════════════════════════════════════
determinante = np.linalg.det(A)

separador("PASO 3 · Verificación del determinante")
print(f"\n   det(A) = {determinante:.4f}")

if abs(determinante) < 1e-10:
    print("\n   ✗ El sistema NO tiene solución única (det ≈ 0).")
    print("     El sistema puede ser incompatible o dependiente.")
    exit()
else:
    print("   ✓ El sistema tiene solución única (det ≠ 0).\n")

# ════════════════════════════════════════════════════════════
# PASO 5: Resolver el sistema con np.linalg.solve()
#
#   np.linalg.solve(A, B) resuelve internamente usando
#   factorización LU con pivoteo parcial, equivalente a
#   la eliminación gaussiana optimizada.
#
#   Devuelve el vector x = [x, y, z] que satisface Ax = B.
# ════════════════════════════════════════════════════════════
solucion = np.linalg.solve(A, B)

# Extraemos cada incógnita por separado para mayor claridad
x = solucion[0]
y = solucion[1]
z = solucion[2]

separador("PASO 4 · Solución del sistema")
print(f"""
   Usando np.linalg.solve(A, B):

   ┌─────────────────────────┐
   │   x = {x:>10.6f}       │
   │   y = {y:>10.6f}       │
   │   z = {z:>10.6f}       │
   └─────────────────────────┘
""")

# ════════════════════════════════════════════════════════════
# PASO 6: Verificación — sustituir en las ecuaciones originales
#
#   Si la solución es correcta, cada ecuación debe dar
#   exactamente el valor del término independiente.
#   También calculamos el residuo: r = B - A·x
#   Una solución perfecta da residuo ≈ 0.
# ════════════════════════════════════════════════════════════
separador("PASO 5 · Verificación de la solución")

# Calcular el lado izquierdo de cada ecuación
eq1 = 2*x + 1*y + 1*z
eq2 = 1*x + 3*y + 2*z
eq3 = 3*x + 2*y + 3*z

print(f"""
   Sustituyendo x={x:.4f}, y={y:.4f}, z={z:.4f}:

   Ec. 1: 2({x:.4f}) + ({y:.4f}) + ({z:.4f})  = {eq1:.6f}   [esperado:  1]
   Ec. 2:  ({x:.4f}) + 3({y:.4f}) + 2({z:.4f}) = {eq2:.6f}  [esperado: 12]
   Ec. 3: 3({x:.4f}) + 2({y:.4f}) + 3({z:.4f}) = {eq3:.6f}  [esperado: 13]
""")

# Verificación automática usando el residuo vectorial
residuo = np.linalg.norm(B - np.dot(A, solucion))
print(f"   Norma del residuo ‖B - Ax‖ = {residuo:.2e}")
if residuo < 1e-10:
    print("   ✓ Verificación superada: solución exacta.\n")
else:
    print("   ⚠ Hay un error numérico significativo.\n")

# ════════════════════════════════════════════════════════════
# PASO 7: Resumen final en pantalla
# ════════════════════════════════════════════════════════════
separador("RESULTADO FINAL")
print(f"""
   Sistema resuelto:
       2x +  y +  z =  1
        x + 3y + 2z = 12
       3x + 2y + 3z = 13

   Solución:
       x = {x:.6f}
       y = {y:.6f}
       z = {z:.6f}

   Interpretación:
       Las corrientes / variables del sistema son:
       x ≈ {x:.4f}
       y ≈ {y:.4f}
       z ≈ {z:.4f}
""")
separador()
print("  Código desarrollado con NumPy · Google Colab")
print("═" * 55)
