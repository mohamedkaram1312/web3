import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set the title of the web app
st.title("Sine Wave Plot")

# Generate data for the sine wave
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Sine Wave', color='blue')
plt.title('Sine Wave')
plt.xlabel('Angle [radians]')
plt.ylabel('Sine Value')
plt.legend()
plt.grid()

# Show the plot in the Streamlit app
st.pyplot(plt)

# Optional: Add a description
st.write("This plot shows the sine function from 0 to 2π.")
