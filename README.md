# Hydraulic-System-PINN-Digital-Twin
# 3D Physics-Informed Neural Network (PINN) for Internal Pipe Flow Simulation

This repository contains a complete, high-fidelity **Physics-Informed Neural Network (PINN)** framework developed in Python to simulate steady-state, incompressible 3D fluid dynamics through a complex pipe bend geometry. 

Instead of traditional, computationally heavy Finite Element Methods (FEM) or Computational Fluid Dynamics (CFD) solvers, this pipeline uses deep learning to solve the governing **3D Navier-Stokes equations** directly on an unstructured point cloud extracted from a physical CAD mesh.

---

## 🚀 Key Framework Features

* **Mesh-to-Neural Pipeline**: Uses `meshio` to parse unstructured grids directly from standard `.msh` files, mapping boundary regions (`Inlet`, `Outlet`, `Walls`, `Fluid`) to exact coordinates.
* **Geometric Proportions Preservation**: Employs global maximum-value scaling normalization to compress coordinates into an optimized range for network convergence without altering the aspect ratio or distorting the pipe geometry.
* **Fixed Anchor Point Training**: Configures the DeepXDE geometry context onto static anchor coordinates (`num_domain=0`, `num_boundary=0`), forcing the network to calculate physical losses strictly within the computational domain boundaries.
* **Point-to-Point Exact Boundary Enforcements**: Implements `PointSetBC` tensors to rigidly clamp physical parameters to discrete coordinate indices (e.g., vertical inlet velocity profiles and precise wall no-slip conditions).
* **Numerically Stable Dynamic Loss Scaling**: Features a custom `SoftmaxAdaptiveWeights` runtime callback. It dynamically balances the trade-offs between momentum residuals, mass conservation, and boundary constraints every 20 epochs using shifted, overflow-protected softmax updates.

---

## 🧬 Governing Physics: 3D Incompressible Navier-Stokes

The neural network is trained by minimizing the residuals of the steady-state, incompressible Navier-Stokes equations over the spatial domain $\mathbf{x} = (x, y, z)$:

1. **Mass Conservation (Continuity Equation):**
   $$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

2. **Momentum Conservation (X, Y, and Z axes):**
   $$(\mathbf{u} \cdot \nabla)\mathbf{u} + \nabla p - \nu \nabla^2 \mathbf{u} = 0$$

Where $\mathbf{u} = (u, v, w)$ is the velocity vector field, $p$ is the kinematic pressure, and $\nu = 0.001$ is the fluid's kinematic viscosity.

---

## 🛠️ Project Structure & Architecture

### **Network Configurations**
* **Type:** Fully Connected Neural Network (`dde.nn.FNN`)
* **Layer Depth:** 3 Inputs $(x,y,z) \rightarrow$ [64, 64, 64] Hidden Units $\rightarrow$ 4 Outputs $(u, v, w, p)$
* **Activation:** Hyperbolic Tangent (`tanh`)
* **Initialization:** Glorot Uniform Distribution

### **Boundary Conditions Configuration**
* **Inlet Bounds:** Controlled vector inputs ($u=0.0$, $v=0.0$, $w=1.0$) pushing fluid straight up the vertical axis.
* **Wall Bounds:** Strict No-Slip Boundary Conditions ($u=0.0$, $v=0.0$, $w=0.0$).
* **Outlet Bounds:** Zero relative pressure threshold ($p=0.0$).

---

## 💻 Prerequisites & Setup

Ensure you have Python 3.8+ installed along with the required libraries. You can install all dependencies via `pip`:

```bash
pip install numpy meshio deepxde scipy matplotlib
```

### Running the Pipeline
1. Place your mesh file (`pipe_bend2.msh`) in the root directory.
2. Execute the primary training script:
   ```bash
   python train_pinn.py
   ```
3. The framework will initialize the deep learning backend, compile the physics graphs, and optimize using the Adam optimizer alongside your custom dynamic softmax callback.

---

## 📊 Post-Processing & Visualization

The repository contains dual post-processing workflows to verify whether the physics equations are being successfully satisfied:

1. **VTK Export for ParaView:** Saves predictions back onto the original unscaled finite element mesh formatting (`pipe_bend_predictions.vtk`) to check pressure fields, velocity magnitudes, and streamlines.
2. **Dynamic Particle Animation:** Generates a real-scale, boundary-masked 2D midplane slice (`pipe_yz_flow_simulation.gif`) using `RegularGridInterpolator` and `KDTree` lookups to visualize moving velocity vector arrows tracking internal streams.

