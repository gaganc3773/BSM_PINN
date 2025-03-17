# **Black-Scholes-Merton Model using Physics-Informed Neural Networks (PINNs)**
## **Overview**
This project applies **Physics-Informed Neural Networks (PINNs)** to solve the **Black-Scholes-Merton (BSM)** equation for pricing **American options**. PINNs enforce the BSM **partial differential equation (PDE)**, boundary conditions, and early exercise constraints, providing a **neural network-driven alternative** to traditional numerical solvers.

---

## **Black-Scholes-Merton Equation**
The **BSM equation** governing option pricing is given by:

$\frac{\partial V}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - rV = 0$

Where:
- **$V(S, t)$** → Option price  
- **$\sigma$** → Volatility  
- **$S$** → Underlying asset price  
- **$r$** → Risk-free interest rate  

For **European options**, the terminal condition at expiration is:
- **Call Option:** $V(S, T) = \max(S - K, 0)$  
- **Put Option:** $V(S, T) = \max(K - S, 0)$  

For **American options**, the solution must satisfy an additional **early exercise constraint**.

---

## **PINN Architecture**
Our **PINN architecture** is designed to approximate the **BSM solution** while enforcing constraints.

### **$1.$ Data Collocation (Training Points Selection)**
- Generate **collocation points** $(S, t)$ where the BSM PDE is enforced.
- The dataset consists of:
  - **Interior points** $(S, t)$ for PDE enforcement.
  - **Boundary points** for known constraints.
  - **Initial condition points** defining the option payoff at $t=0$.

---

### **$2.$ Forward Propagation (Neural Network Prediction)**
- **Input:** $(S, t)$  
- **Network Structure:**
  - **Input layer:** $(S, t)$  
  - **Multiple hidden layers** with **tanh activation**  
  - **Output layer:** $V_{\theta}(S, t)$ (predicted option price)  

The neural network approximation is:  
$V_{\theta}(S, t) = \text{NN}(S, t; \theta)$  

---

### **$3.$ Loss Computation**
#### **Residual Loss (PDE Loss)**
To satisfy the BSM equation, we minimize:  

**PDE Loss:**  
```latex
\mathcal{L}_{\text{PDE}} = \bigg( \frac{\partial V_{\theta}}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 V_{\theta}}{\partial S^2} + r S \frac{\partial V_{\theta}}{\partial S} - rV_{\theta} \bigg)^2
```
Boundary conditions are enforced:
- **European Option:** $V(S, T) = \max(S - K, 0)$  
- **American Option:** Additional early exercise constraint applies.

#### **Early Exercise Condition (American Option Constraint)**
$V(S, t) \geq P(S)$ where $P(S)$ is the **intrinsic value**.

#### **Total Loss Function**
The **total loss function** is a weighted sum of all components:  
```latex
$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{PDE}} + \lambda_2 \mathcal{L}_{\text{boundary}} + \lambda_3 \mathcal{L}_{\text{exercise}}$
```
---

### **$4.$ Backward Propagation & Optimization**
- Compute gradients using **automatic differentiation (AD)** in PyTorch.
- Use **Adam optimizer** to update weights $\theta$.
- The optimization step follows:  
  $\theta = \theta - \eta \frac{\partial \mathcal{L}}{\partial \theta}$  
  where **$\eta$** is the learning rate.

---

### **$5.$ Training Process**
- Train until the **loss converges**.
- Monitor **loss components**: PDE, boundary, and exercise condition loss.
- Validate against the **Black-Scholes closed-form solution**.

---

## **Features**
- Solves the **BSM PDE** for American options using **PINNs**  
- Uses **automatic differentiation** to impose PDE constraints  
- Implements **early exercise conditions** for American options  
- Compares **PINN predictions** with analytical solutions  

---

## **References**
- **Andreas Luoskos** et al., *"Physics-Informed Neural Networks and Option Pricing"*



