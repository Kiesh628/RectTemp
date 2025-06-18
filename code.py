import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

SIMULATION_TIME = 150  

PLATE_LENGTH = 50   
PLATE_WIDTH = 50    
DX = 0.01           

ALPHA = 1.9e-5      


INITIAL_TEMP = 20.0  
HOT_SIDE_TEMP = 100.0 
COLD_SIDE_TEMP = 20.0 

DT = (DX**2) / (4 * ALPHA)
NUM_STEPS = int(SIMULATION_TIME / DT)

def calculate_next_step(T):
    T_new = T.copy()

    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + ALPHA * DT / DX**2 * \
        (T[2:, 1:-1] + T[:-2, 1:-1] + T[1:-1, 2:] + T[1:-1, :-2] - 4 * T[1:-1, 1:-1])

    return T_new



if __name__ == "__main__":
    print("--- 2D Transient Heat Conduction Simulator ---")
    print(f"Plate size: {PLATE_WIDTH}x{PLATE_LENGTH} nodes")
    print(f"Grid spacing (dx): {DX} m")
    print(f"Time step (dt): {DT:.4f} s")
    print(f"Total simulation time: {SIMULATION_TIME} s")
    print(f"Number of time steps: {NUM_STEPS}")
    print("---------------------------------------------")


    T = np.full((PLATE_LENGTH, PLATE_WIDTH), INITIAL_TEMP, dtype=float)

    T[0, :] = HOT_SIDE_TEMP   
    T[-1, :] = COLD_SIDE_TEMP  
    T[:, 0] = COLD_SIDE_TEMP   
    T[:, -1] = COLD_SIDE_TEMP  


    norm = Normalize(vmin=COLD_SIDE_TEMP, vmax=HOT_SIDE_TEMP)


    plt.figure(figsize=(8, 6))
    plt.imshow(T, cmap='hot', origin='lower', norm=norm)
    plt.colorbar(label='Temperature (°C)')
    plt.title('Initial Temperature Distribution')
    plt.xlabel('X-axis Node')
    plt.ylabel('Y-axis Node')
    plt.show()


    print("Running simulation to find the final state...")
    T_final = T.copy()
    for i in range(NUM_STEPS):
        T_final = calculate_next_step(T_final)
    print("Simulation complete.")


    plt.figure(figsize=(8, 6))
    plt.imshow(T_final, cmap='hot', origin='lower', norm=norm)
    plt.colorbar(label='Temperature (°C)')
    plt.title(f'Final Temperature Distribution (after {SIMULATION_TIME}s)')
    plt.xlabel('X-axis Node')
    plt.ylabel('Y-axis Node')
    plt.show()


    print("Preparing animation...")
    fig, ax = plt.subplots(figsize=(8, 6))


    T_anim = T.copy()

    im = ax.imshow(T_anim, cmap='hot', origin='lower', norm=norm, animated=True)
    cb = fig.colorbar(im, label='Temperature (°C)')
    title = ax.set_title("Time: 0.00s")
    ax.set_xlabel('X-axis Node')
    ax.set_ylabel('Y-axis Node')

    def animate(frame):
        global T_anim
        steps_per_frame = 20
        for _ in range(steps_per_frame):
            T_anim = calculate_next_step(T_anim)
        
        im.set_array(T_anim)
        current_time = frame * steps_per_frame * DT
        title.set_text(f"Time: {current_time:.2f}s")
        return im, title,

    num_frames = NUM_STEPS // 20
    ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=20, blit=True)
    plt.show()
