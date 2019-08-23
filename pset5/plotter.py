import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

# Set Endianness. If using a BG/Q Vesta big endian binary file, set to "True"
BigEndian = False

# Input file name
fname = "nbody.dat"

print("loading input file ", fname," ...")

# Opens input file
f = open(fname, 'rb')

# Reads header info (nBodies, nIters)
first_line = f.read(8) # We read two 4 bytes integers
line = np.frombuffer(first_line, dtype=np.int32)
if(BigEndian):
    line = line.byteswap()

nBodies = int(line[0])
timesteps = int(line[1])
print("nBodies in file = ", nBodies, " Timesteps in file = ", timesteps)

# Adjusts marker size based on number of bodies in problem
marker = 1.0
if (nBodies > 100 ):
    marker = 0.5
if (nBodies > 1000 ):
    marker = 0.25
if (nBodies > 5000 ):
    marker = 0.1

# Allocations array to hold a timestep
arr = np.empty(dtype=np.float64, shape=(nBodies,3))

# Reads initial conditions
for i in range(nBodies):
    line = f.read(24)
    body = np.frombuffer(line, dtype=np.float64)
    if(BigEndian):
        body = body.byteswap()
    arr[i,:] = body

# Create a 3D plot and initialize it with initial particle states
fig, ax = plt.subplots()
ax = p3.Axes3D(fig)
points, = [],

if (nBodies >= 10000 ):
    # If we have a lot of bodies, only plot pixels
    points, = ax.plot3D(arr[:,0], arr[:,1], arr[:,2], 'w,')
else:
    # For fewer bodies, use a larger marker size
    points, = ax.plot3D(arr[:,0], arr[:,1], arr[:,2], 'wo', markersize=marker)

# Plot Info
# NOTE: you may want to adjust the boundaries based on your body intialization scheme
bounds = 2.0
ax.set_ylim(-bounds, bounds)
ax.set_xlim(-bounds, bounds)
ax.set_zlim3d(-bounds, bounds)
ax.set_facecolor('xkcd:black')
plt.axis('off')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Function that will be called for each frame of the animation
def update(data):
    update.t += 1
    print("Processing Time Step ", update.t)
    # Reads a set of bodies into an array
    arr = np.empty(dtype=np.float64, shape=(nBodies,3))
    for i in range(nBodies):
        line = f.read(24)
        body = np.frombuffer(line, dtype=np.float64)
        if(BigEndian):
            body = body.byteswap()
        arr[i,:] = body

    points.set_xdata(arr[:,0])
    points.set_ydata(arr[:,1])
    points.set_3d_properties(arr[:,2]) # No set_zdata, se we use this

    return points,

update.t = -1

# Generate the animation
ani = animation.FuncAnimation(fig, update, timesteps-2)

# Save .mp4 of the animation
# NOTE: Bitrate, resolution, and dpi may need to be adjusted
ani.save('nbody_simulation.mp4', fps=60, bitrate=100000, extra_args=["-s", "2560x1440"], dpi=400)
#plt.show()
