import matplotlib.pyplot as plt

from dolfin import *
from dolfin.function.argument import Argument

import math
import numpy as np

# create mesh and define function spaces
nx: int = 3
ny: int = 3

print('name', 'type', 'value')

mesh: UnitSquareMesh = UnitSquareMesh(nx, ny)
# print('mesh\t', type(mesh), '\t', mesh)

Va: FunctionSpace = FunctionSpace(mesh, 'Lagrange', 1)  # a为什么用1？
# print('Va\t', type(Va), '\t', Va)

a_trial: Argument = TrialFunction(Va)
# print('a_trial\t', type(a_trial), '\t', a_trial)
a_test: Argument = TestFunction(Va)

a: Function = interpolate(Constant(4.), Va)
print('a\t', type(a), '\t', ' ')
# print('a.vector()\t', type(a.vector()), '\t', a.vector())
# print('a.vector().get_local()\t', type(
#     a.vector().get_local()), '\t', a.vector().get_local())

R_equ: Form = inner(nabla_grad(a_trial), nabla_grad(a_test)) * dx
# print('R_equ\t', type(R_equ), '\t', R_equ)

W: Matrix = assemble(R_equ)
# print('W\t', type(W), '\t', W)

# print('W.array()\t', type(W.array()), '\t', W.array())

g = Vector()
# print('g\t', type(g))

b: Function = Function(Va)  # 生成默认值都是0的函数
print('b.vector()', type(b.vector()), b.vector().get_local())
