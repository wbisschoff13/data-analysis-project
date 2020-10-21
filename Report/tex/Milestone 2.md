---
title: "Habits"
author: John Doe
date: March 22, 2005
---

### Introduction 
A DC motor is to be characterized and a mathematical model is to be derived. 

### First Order Model


### The Method
Calculate the armature resistance $R$ by measuring the armature current $I$ when supplying voltage $V$ to a short-circuit and measuring the armature voltage $E$ when supplying voltage to the open circuit.

Applying an AC voltage and determining the armature voltage and current allows for the calculation of the armature impedance $Z$. The inductance $L$ can be calculated from the reactance $X$ of the impedance.

The back-EMF constant $K_e$ can be determined from $$K_e = \frac{\Delta E}{\Delta\omega}, $$ and the torque constant $K_t$ can be determined as $K_t = K_e$. 

The damping coefficient $b$ can be determined from $$b\omega = K_tI_a, $$ which is derived from $$T=K_ti$$ and $$T = J\dot\omega +b\omega, $$ while keeping the system at a steady state so that $\dot\omega = 0$.

The motor inertia $J$ can be calculated from $$J = -\frac{T}{\frac{dw}{dt}}.$$


### Results

The resistance $R = 10,42\Omega$ can be calculated from Table 1. 

| $V_f$ | $E_a$ | $I_a$ |
|-------|-------|-------|
| 0     | 19.10 | 1,90  |
| 4     | 23,03 | 2,25  |
| 8     | 26,19 | 2,58  |
| 12    | 29,82 | 2,81  |
| 16    | 34,16 | 3,13  |
| 20    | 37,67 | 3,48  |

The back-EMF constant and the torque constants can be calculated as $K_t = K_e = 0.728$, the damping coefficient can be calculated as $b = 0.014$. The inductance $L = 0.15$.


| Ea  | Ia    | w           |
|-----|-------|-------------|
| 24  | 1,186 | 12,15 |
| 38  | 1,408 | 28,48 |
| 55  | 1,605 | 49,95 |
| 67  | 1,67  | 66,71 |
| 76  | 1,705 | 79,27 |
| 84  | 1,72  | 89,12 |
| 92  | 1,738 | 100,16 |
| 96  | 1,747 | 109,01 |
| 105 | 1,752 | 120,32 |
| 115 | 1,78  | 135,40 |
| 128 | 1,76  | 155,72 |

The inertia $J = 0,022$.

The transfer function $P(s)$ can be determined as

$$P(s)=\frac{\omega(s)}{V(s)}=\frac{K_t}{(Js+b)(Ls+R)+K_eK_t}.$$