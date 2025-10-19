import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

from bsm_pinn.model import PINN
from bsm_pinn.train import train_pinn, make_collocation_points


def main():
    st.set_page_config(page_title="BSM PINN", layout="wide")
    st.title("Black–Scholes–Merton PINN Web App")

    with st.sidebar:
        st.header("Training Settings")
        n_epochs = st.number_input("Epochs", 1, 10000, 200, step=50)
        n_f = st.number_input("Collocation points", 1000, 200000, 20000, step=1000)
        lr = st.number_input("Learning rate", 1e-5, 1e-1, 5e-3, step=1e-3, format="%f")
        t_max = st.number_input("Max time T", 0.1, 5.0, 3.0, step=0.1)
        s_max = st.number_input("Max stock S_max", 10.0, 2000.0, 500.0, step=10.0)
        sigma = st.number_input("Volatility σ", 0.01, 2.0, 0.4, step=0.01)
        r = st.number_input("Risk-free rate r", 0.0, 0.5, 0.03, step=0.005)
        activation = st.selectbox("Activation", ["tanh", "relu", "gelu"], index=0)
        hidden = st.text_input("Hidden sizes (comma)", value="50,50")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.caption(f"Device: {device}")

    hidden_sizes = tuple(int(h.strip()) for h in hidden.split(",") if h.strip()) or (50, 50)

    if st.button("Train PINN", type="primary"):
        with st.spinner("Training..."):
            model = PINN(activation=activation, hidden_sizes=hidden_sizes)
            losses, model = train_pinn(
                model,
                n_epochs=n_epochs,
                n_f=n_f,
                lr=lr,
                t_max=t_max,
                s_max=s_max,
                sigma=sigma,
                r=r,
                device=device,
            )
        st.success("Training complete!")
        st.line_chart(losses, height=220)

        # Simple prediction slice at fixed t
        st.header("Prediction Slice")
        t_slice = st.slider("t for slice", 0.0, float(t_max), min(float(t_max), 1.0), step=0.05)
        s_vals = torch.linspace(0, s_max, 200).reshape(-1, 1)
        t_vals = torch.full_like(s_vals, t_slice)
        X = torch.cat([t_vals, s_vals], dim=1).to(device)
        with torch.no_grad():
            y = model(X).cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(s_vals.numpy(), y, label="PINN")
        ax.set_xlabel("S")
        ax.set_ylabel("V(S,t)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Export
        if st.button("Download predictions CSV"):
            arr = np.hstack([s_vals.numpy(), y])
            st.download_button(
                label="Download CSV",
                data="\n".join(",".join(map(str, row)) for row in arr),
                file_name="predictions.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
