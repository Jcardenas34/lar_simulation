

from .create_objects import *
from .calculations import *

def simulate_solid(N, runtime, dt, dimension, material, lattice_spacing, box_len):
    t = 0.0  # starting time
    atoms = []  ## list that will contain all the particles to be displayed and thier attributes
    # atoms = np.array(sphere(pos= vector(0,0,0), radius= radius*L, color= color.red),dtype=object)

    v = np.zeros(
        (N, 3)
    )  ## returns an N row by 3 column matrix to store the 3 components of VELOCITY
    mean_sqr_array = []
    # time_array = np.arange(0.0, runtime + dt, dt)

    #### Simulation starts here =====
    generate_solid_camera(dimension, lattice_spacing, box_len)
    positions, atoms = create_solid(dimension, lattice_spacing, box_len)

    print("Simulating N={} molecules".format(N))
    while t < runtime:
        #     #vp.rate(2000)

            # r[r > ((dimension-1)*lat_c)/2 + HL] -= L                         # periodic boundary conditions
            # r[r < ((dimension-1)*lat_c)/2 - HL] += L

        a = nbody(positions, N, material, box_len)  # calculate the acceleration due to current arrangement

        positions, v, atoms, mean_sqr_array = new_position_velocity(
            positions, v, a, atoms, material, mean_sqr_array, N, dt, box_len
        )

        t += dt
    print("=== End of simulation ===\n")





def simulate_liquid(N, runtime, dt, radius, L):
    # dimension = 3                       # dimension of the particle lattice we start with, for example: 2x2x2
    L = 17  # cube size
    # L = 10                              # cube size
    HL = L / 2  # Half length of the box
    radius = 0.02
    # N = dimension**3 + 3*(dimension*((dimension - 1)**2))            # number of atoms in FCC dimension
    N = 110  # can un comment this to simulate any number of particles

    lat_c = 1.77

    t = 0.0  # starting time
    # dt = 0.004                          # increment of time, same as dt
    runtime = 30  # in seconds
    floor_atoms = 10

    atoms = []  # list that will contain all the particles to be displayed and thier attributes
    r = np.zeros(
        (N, 3)
    )  ## returns an N row by 3 column matrix to store the 3 components of POSITION
    v = np.zeros(
        (N, 3)
    )  ## returns an N row by 3 column matrix to store the 3 components of VELOCITY
    a = np.zeros((N, 3))
    mean_sqr_array = []
    time_array = np.arange(0.0, runtime, dt)

    scene = canvas(
        title="Argon Liquid Simulation",
        center=vector(L / 2, L / 2, L / 2),
        background=color.black,
        width=1000,
        height=700,
    )
    scene.autoscale = False
    box(pos=vector(HL, HL, HL), length=L, height=L, width=L, opacity=0.3)

    # r, v, atoms = create_liquid(r, v, atoms, radius, N, L)

    r, v, atoms = create_glob(r, v, atoms, N, radius, L)

    # r, v, atoms = create_floor(r,v,atoms,floor_atoms,radius,L,lat_c)

    print("Simulating N={} molecules".format(N))
    print("=== Now starting simulation ===")
    while t < runtime:  ## can set the argument to a number to make it run indefinitly
        # vp.rate(2000) # puts a limit of x amount of frames per second
        vp.rate(10000)  # puts a limit of x amount of frames per second
        r[r > L] -= L  # periodic bc
        r[r < 0.0] += L

        a = nbody(r, N, args.simulate, L)
        r, v, atoms, mean_sqr_array = new_position_velocity(
            r, v, a, atoms, mean_sqr_array, N, dt, L
        )

        t += dt
    print("=== End of simulation ===\n")

    # print("=== Creating Mean Square Distance plot for Liquid ===")
    # plt.plot(time_array,mean_sqr_array, label = "Particles = {:}".format(N))
    # plt.legend()
    # plt.title("Liquid Argon: Mean Square Distance vs Time")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Mean sqr distance")
    # print("=== Click x on plot to continue to simulation ===")
    # plt.show()
    # plt.close()

    # print("=== Making Pair Correlation Function ===")
    # range_array, pair_corr_array = pair_correlation_function_liquid(r,HL,N)
    # plt.plot(range_array, pair_corr_array, label = "Particles = {:}".format(N))
    # plt.legend()
    # plt.title("Pair Correlation Function: Liquid")
    # plt.xlabel("Distance")
    # plt.ylabel("g(r)")
    # print("=== Press control+c at terminal to end the program ===")
    # plt.show()
    # plt.close()


"""
#### Code for 2D lattice below ======================================================
"""


def simulate_2D_lattice(N, runtime, dt, dimension, lat_c, radius, L, HL):
    radius = 0.02
    lat_c = 1.225  # lattice spacing for argon:
    #  W/12: .98 ,  W/48 adjusted: 1.4  W/book and our specific 1.72 W/book   from book/ 1.225
    dimension = 5  # dimension of the particle lattice we start with, for example: 2x2x2
    L = dimension * lat_c  # + .5*lat_c                           # container size
    HL = L / 2
    Num = dimension**2  # for square, square faced lattice

    atoms = []  # list that will contain all the particles to be displayed and thier attributes
    r = np.zeros(
        (Num, 3)
    )  ## returns an N row by 3 column matrix to store the 3 components of POSITION
    v = np.zeros(
        (Num, 3)
    )  ## returns an N row by 3 column matrix to store the 3 components of VELOCITY

    t = 0.0  # starting time
    dt = 0.004  # increment of time, same as dt
    runtime = 60  # in seconds

    scene2 = canvas(
        title="Argon 2D Lattice Simulation",
        center=vector(
            float(((dimension - 1) * lat_c) / 2),
            float(((dimension - 1) * lat_c) / 2),
            float(((dimension - 1) * lat_c) / 2),
        ),
        background=color.black,
        width=1000,
        height=700,
    )

    box(
        pos=vector(
            float(((dimension - 1) * lat_c) / 2),
            float(((dimension - 1) * lat_c) / 2),
            float(((dimension - 1) * lat_c) / 2),
        ),
        length=L,
        height=L,
        width=L,
        opacity=0.3,
    )

    r, v, atoms = create_2D_lattice(r, atoms, dimension, lat_c, radius, Num, L)

    mean_sqr_array = []
    print("Simulating N={} molecules".format(Num))
    print("=== Now starting simulation ===")
    while t < runtime:  ## can set the argument to a number to make it run indefinitly
        vp.rate(2000)  # puts a limit of x amount of frames per second

        r[r > ((dimension - 1) * lat_c) / 2 + HL] -= L  # periodic bc
        r[r < ((dimension - 1) * lat_c) / 2 - HL] += L

        a = nbody_2D_lattice(r, Num, L)
        r, v, atoms, mean_sqr_array = new_position_velocity(
            r, v, a, atoms, mean_sqr_array, Num, dt, L
        )

        t += dt

    print("=== End of simulation ===")
    print("=== Press control+c at terminal to end the program ===")
