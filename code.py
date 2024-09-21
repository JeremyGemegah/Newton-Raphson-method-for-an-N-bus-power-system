import numpy as np

def get_admittance_matrix_rectangular(n):#def with n being size of matrix hehe
    Y = np.zeros((n, n), dtype=complex)#np is numpy library this part initializes a zero matrix
    print("\nInput the admittance matrix (G + jB) for each element in the format (G B):")
    print("For example, 4-j7 will be inputed as:4 -7")
    for i in range(n):#this initiates a loop that will iterate n times where i will take vakues from 0 to n-1 for row
        for j in range(n):#nested loop iterating n times for each row used to process each column
            while True:
                try:
                    entry = input(f"Enter Y[{i + 1}][{j + 1}] in the format (G B): ").strip()#inserts the values into the message and strip to remove trailspaceerror
                    G, B = map(float, entry.split())#splits into a list of substrings and map applies the float function converting thesubstrings to floa
                    Y[i, j] = G + 1j * B# creates a complex number
                    break
                except ValueError:
                    print("\nPlease input the value in the right format\n")
            
    return Y#returns admittance matrix to caller of function
#the functions of this functions are same as above juat polar so hehe
def get_admittance_matrix_polar(n):
    Y = np.zeros((n, n), dtype=complex)
    print("\nInput the admittance matrix |Y|∠θ for each element in the format (Magnitude, followed by angle):")
    for i in range(n):
        for j in range(n):
                while True:
                    try:
                        magnitude = float(input(f"Enter the magnitude of Y[{i + 1}][{j + 1}]: ").strip())
                        break
                    except ValueError:
                        print("\nIVALID INPUT\nPlease input the value in the right format\n")
                        continue

                while True:  
                    try:      
                        angle = float(input(f"Enter the angle of Y[{i + 1}][{j + 1}] in degrees: ").strip())
                        break
                    except ValueError:
                        print("\nIVALID INPUT\nPlease input the value in the right format\n")
                        continue
                Y[i, j] = magnitude * np.exp(1j * np.radians(angle))#converts from rad to deg since numpy uses rad in calculation
                break
            
    return Y

def display_matrix(matrix, title):# function that takes 2 paramaters as you an see hehe
    print(f"\n{title}:")
    for row in matrix:#loops through each row
        print(" ".join(f"{value.real:.4f} + j{value.imag:.4f}".rjust(20) for value in row))#generates a string that joins each formatted element with a space and extracts to 4d.p
#also rigt justifies within a space of 20 characters for neatness
def display_real_matrix(matrix, title):
    print(f"\n{title}:")
    for row in matrix:
        print(" ".join(f"{value:.6f}".rjust(15) for value in row))

def display_vector(vector, title):#displays a one dimentional vector with its elements formatted to 6dp
    print(f"\n{title}:")
    for i, value in enumerate(vector):
        print(f"Element {i + 1}: {value:.6f}")

def get_bus_data(n):#here n is no of buses
    bus_data = []#initialize empty bus
    for i in range(n):#just a boring loop as usual
        while True:
            try:
                print(f"\nEnter data for Bus {i + 1}:")#displays bus number starting from 1 instead of 0
                bus_type = int(input("Enter bus type \n1) Slack\n2)PQ\n3)PV\n👉 ").strip())
                
                if bus_type == 1:
                    while True:
                        try:
                            V = float(input("Enter the voltage magnitude for the Slack bus (per unit): ").strip())
                            theta = float(input("Enter the voltage angle for the Slack bus (degrees): ").strip())
                            bus_data.append({'type': 'slack', 'V': V, 'theta': theta, 'P': 0, 'Q': 0})#store slack bus data with p and q 0 sine slack bus doesnt store data
                            break
                        except ValueError:
                            print("\nINVALID INPUT\nPlease try again")

                elif bus_type == 2:
                    while True:
                        try:
                            P = float(input("Enter specified active power (P) (per unit): "))
                            Q = float(input("Enter specified reactive power (Q) (per unit): "))
                            V = 1.0  # Flat start voltage magnitude
                            theta = 0.0  # Flat start voltage angle
                            bus_data.append({'type': 'PQ', 'P': P, 'Q': Q, 'V': V, 'theta': theta})
                            break
                        except ValueError:
                            print("\nINVALID INPUT\nPlease try again")
                elif bus_type == 3:
                    while True:
                        try:
                            P = float(input("Enter specified active power (P) (per unit): "))
                            V = float(input("Enter specified voltage magnitude (per unit): "))
                            theta = 0.0  # Flat start voltage angle
                            bus_data.append({'type': 'PV', 'P': P, 'Q': 0, 'V': V, 'theta': theta})
                            break
                        except ValueError:
                            print("\nINVALID INPUT\nPlease try again")
                else:
                    print("\nINVALID INPUT\nPlease try again")
                    continue
                break
            except ValueError:
                print("\nINVALID INPUT\nPlease try again")
            
    return bus_data



def random_bus_data(n):
    bus_data = []
    
    # Control the randomness within realistic constraints
    P_load_base = 0.5  # Base real power load
    Q_load_base = 0.3  # Base reactive power load
    P_load_variation = 0.2  # Variation in real power load
    Q_load_variation = 0.15  # Variation in reactive power load

    for i in range(n):
        if i == 0:
            # Slack Bus Initialization
            bus_data.append({'type': 'slack', 'V': 1.0, 'theta': 0.0, 'P': 0, 'Q': 0})
            print(f"Bus {i + 1}: Slack Bus (V = {1.0}, θ = {0.0} degrees)")
        else:
            bus_type = np.random.choice(['PQ', 'PV'])
            if bus_type == "PQ":
                # Realistic load values with controlled randomness
                P = P_load_base + np.random.uniform(-P_load_variation, P_load_variation)
                Q = Q_load_base + np.random.uniform(-Q_load_variation, Q_load_variation)
                V = np.random.uniform(0.95, 1.05)  # Voltage magnitude close to 1.0 pu
                theta = np.radians(np.random.uniform(-5, 5))  # Small variation in angle (in radians)
                bus_data.append({'type': 'PQ', 'P': P, 'Q': Q, 'V': V, 'theta': theta})
                print(f"Bus {i + 1}: PQ Bus (P = {P:.4f}, Q = {Q:.4f}, V = {V:.2f}, θ = {np.degrees(theta):.2f} degrees)")
            elif bus_type == "PV":
                # Realistic generation values with controlled randomness
                P = P_load_base + np.random.uniform(-P_load_variation, P_load_variation)
                V = np.random.uniform(0.95, 1.05)  # Voltage magnitude slightly varied around 1.0 pu
                Q_min = -0.5  # Lower limit for reactive power
                Q_max = 0.5  # Upper limit for reactive power
                Q = 0.0  # Initially, PV bus starts with Q = 0

                # Precondition for reactive power limits
                Q_condition = np.random.choice([True, False])
                if Q_condition:  # Simulate condition where Q hits its limits
                    Q = np.random.choice([Q_min, Q_max])
                    bus_type = "PQ"  # Convert to PQ bus if Q limit is hit

                theta = np.radians(np.random.uniform(-5, 5))  # Small variation in angle (in radians)
                bus_data.append({'type': bus_type, 'P': P, 'Q': Q, 'V': V, 'theta': theta})
                print(f"Bus {i + 1}: {bus_type} Bus (P = {P:.4f}, Q = {Q:.4f}, V = {V:.2f}, θ = {np.degrees(theta):.2f} degrees)")
    
    return bus_data

def random_admittance_matrix(n, sparsity=0.1):
    Y = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal elements: typically larger and realistic
                G = np.random.uniform(3, 7)  # Conductance
                B = np.random.uniform(-25, 14)  # Susceptance
                Y[i, j] = G + 1j * B
            else:
                # Off-diagonal elements: small values reflecting weak coupling
                if np.random.rand() > sparsity:
                    G = np.random.uniform(-2, 7)
                    B = np.random.uniform(6, 12)
                    Y[i, j] = G + 1j * B
                    Y[j, i] = Y[i, j]  # Symmetric matrix
    
    # Ensure diagonal dominance for numerical stability
    for i in range(n):
        off_diag_sum = sum(abs(Y[i, j]) for j in range(n) if j != i)
        if abs(Y[i, i]) <= off_diag_sum:
            # Increase the diagonal element to ensure diagonal dominance
            Y[i, i] += 1.1 * off_diag_sum
    
    return Y



def calculate_power_mismatch(bus_data, Y, V, theta,n):
    P_calc = np.zeros(n)
    Q_calc = np.zeros(n)#p cal and q all initialized to zero

    for i in range(n):
        for j in range(n):
            P_calc[i] += V[i] * V[j] * (Y[i, j].real * np.cos(theta[i] - theta[j]) + Y[i, j].imag * np.sin(theta[i] - theta[j]))
            Q_calc[i] += V[i] * V[j] * (Y[i, j].real * np.sin(theta[i] - theta[j]) - Y[i, j].imag * np.cos(theta[i] - theta[j]))
#these two form computes by summing the contributionsfrom all buses j to i
    P_spec = np.array([bus_data[i]['P'] for i in range(n)])
    Q_spec = np.array([bus_data[i]['Q'] for i in range(n)])#retrieve specified power from busdata which holds details for each bus yessir


    P_mismatch = P_spec - P_calc
    Q_mismatch = Q_spec - Q_calc

    return P_mismatch, Q_mismatch


def compute_jacobian_matrix(Y, V, theta, PQ_indices, PV_PQ_indices,num_PV_PQ,n):

    P_calc = np.zeros(n)
    Q_calc = np.zeros(n)

    for i in range(n):
        for j in range(n):
            P_calc[i] += V[i] * V[j] * (Y[i, j].real * np.cos(theta[i] - theta[j]) + Y[i, j].imag * np.sin(theta[i] - theta[j]))
            Q_calc[i] += V[i] * V[j] * (Y[i, j].real * np.sin(theta[i] - theta[j]) - Y[i, j].imag * np.cos(theta[i] - theta[j]))
#arrays that store calc values based on current v mag,angle and admittance matrix
    J = np.zeros((num_PV_PQ + len(PQ_indices), num_PV_PQ + len(PQ_indices)))

    for row, i in enumerate(PV_PQ_indices):
        for col, j in enumerate(PV_PQ_indices):
            if i == j:#for diagonal elements
                J[row, col] = -Q_calc[i] - (V[i]**2) * Y[i, i].imag #  piwrtangi
                if i in PQ_indices:
                    pq_row = num_PV_PQ + PQ_indices.index(i)
                    J[pq_row, col] = P_calc[i] - (V[i]**2) * Y[i, i].real  #  qiwrtangi
                    J[row, pq_row] = P_calc[i] / V[i] + V[i] * Y[i, i].real   # changes piwrtvi
                    J[pq_row, pq_row] = Q_calc[i] / V[i] - V[i] * Y[i, i].imag  #qiwrtvi
            else:
                J[row, col] = V[i] * V[j] * (Y[i, j].real * np.sin(theta[i] - theta[j]) - Y[i, j].imag * np.cos(theta[i] - theta[j]))  #  p wrt ang for off diagonal
                if i in PQ_indices:
                    pq_row = num_PV_PQ + PQ_indices.index(i)
                    J[pq_row, col] = -V[i] * V[j] * (Y[i, j].real * np.cos(theta[i] - theta[j]) + Y[i, j].imag * np.sin(theta[i] - theta[j]))  # q wrt ang for off diagonal
                if j in PQ_indices:  # Check if j is in PQ_indices before using its index
                    J[row, num_PV_PQ + PQ_indices.index(j)] = V[i] * V[j] * (Y[i, j].real * np.cos(theta[i] - theta[j]) + Y[i, j].imag * np.sin(theta[i] - theta[j]))  # p wrt v for off diagonal
                    if i in PQ_indices:
                        pq_row = num_PV_PQ + PQ_indices.index(i)
                        J[pq_row, num_PV_PQ + PQ_indices.index(j)] = V[i] * (Y[i, j].real * np.sin(theta[i] - theta[j]) - Y[i, j].imag * np.cos(theta[i] - theta[j]))  # q wrt v for off diagonal

    return J


def update_bus_voltages(delta, V, theta, PQ_indices, PV_PQ_indices,num_PV_PQ):#delta is for corrections for v and angle where v and theta are initial values
    for k, i in enumerate(PV_PQ_indices):
        theta[i] += delta[k]#angle is updated here from theory i studied
        if i in PQ_indices:
            V[i] += delta[num_PV_PQ + PQ_indices.index(i)]#v is updated here
    return V, theta#return updated values


def newton_raphson_load_flow(bus_data, Y, tolerance=1e-4, max_iterations=10000):
    ConvergenceReached = False
    n = len(bus_data)
    PV_indices = [i for i in range(n) if bus_data[i]['type'] == 'PV']
    PQ_indices = [i for i in range(n) if bus_data[i]['type'] == 'PQ']
    PV_PQ_indices = PV_indices + PQ_indices
    num_PV_PQ = len(PV_PQ_indices)

    V = np.array([bus_data[i]['V'] for i in range(n)])
    theta = np.array([bus_data[i]['theta'] for i in range(n)])

    first_iteration = True  # Flag to indicate the first iteration

    for iteration in range(max_iterations):
        P_mismatch, Q_mismatch = calculate_power_mismatch(bus_data, Y, V, theta,n)
        mismatch = np.concatenate((P_mismatch[PV_PQ_indices], Q_mismatch[PQ_indices]))

        if np.all(np.abs(mismatch) < tolerance):
            print(f"\nConvergence achieved in {iteration + 1} iterations.")
            break

        J = compute_jacobian_matrix(Y, V, theta,PQ_indices, PV_PQ_indices,num_PV_PQ,n)
        J_inv = np.linalg.inv(J)
        delta = J_inv @ mismatch

        V, theta = update_bus_voltages(delta, V, theta,PQ_indices, PV_PQ_indices,num_PV_PQ)

        if first_iteration:
            first_iteration = False
            print("\n--- Results from the First Iteration ---")
            display_real_matrix(J, "Jacobian Matrix (J)")
            display_real_matrix(J_inv, "Inverse of the Jacobian Matrix (J_inv)")
            display_vector(mismatch, "Power Mismatch Vector")

            print("\nUpdated Voltage Magnitudes and Angles after the First Iteration:")
            ConvergenceReached = True
            for i in range(n):
                print(f"Bus {i + 1}: V = {V[i]:.6f} pu, θ = {np.degrees(theta[i]):.6f} degrees")


    else:

        print("\nMaximum iterations reached without convergence.\n")

        print("Possible Reasons for Non-Convergence:\n")
        print("1. Poor initial guess for voltage magnitudes or angles.")
        print("2. Tolerance level might be too strict, making it difficult to achieve convergence.")
        print("3. Poor network data (e.g., unrealistic bus data or line impedances).")
        print("4. Numerical instability in the Jacobian matrix.")
        print("5. Max iterations reached.\n")

        while True:
            try:
                response = input("Would you like to generate new data? (yes/no))\nChoose NO to view the results of last iteration regardless\nChoose YES to forfeit results and go back to main menu\n ").strip().lower()
                if response != 'yes' and response != 'no':
                    print("\nINVALID INPUT\nPlease try again")
                    continue
                elif response == 'yes':
                    main()
                    print("\nRestarting data generation...")
                    break
                else:
                    break
            except ValueError:
                print("\nINVALID INPUT\nPlease try again")

        

    print("\n--- Results from the Last Iteration ---")
    print("\nFinal Voltages and Angles:")
    for i in range(n):
        print(f"Bus {i + 1}: V = {V[i]:.6f} pu, θ = {np.degrees(theta[i]):.6f} degrees")

    display_real_matrix(J, "Jacobian Matrix (J)")
    display_real_matrix(J_inv, "Inverse of the Jacobian Matrix (J_inv)")
    display_vector(mismatch, "Power Mismatch Vector")

    print("\nUpdated Voltage Magnitudes and Angles after the Last Iteration:")
    for i in range(n):
        print(f"Bus {i + 1}: V = {V[i]:.6f} pu, θ = {np.degrees(theta[i]):.6f} degrees")
    return ConvergenceReached

def main():# firsttttt entry point
    
    
    print("\n\nHELLO, WELCOME TO GROUP 7'S NEWTON RAPHSON METHOD SOLVER FOR LOAD FLOW STUDIES 😊\n")

    while True:
        convergenceReached = False
        try:
            choice = input("HOW DO YOU WISH TO PROVIDE INPUT DATA ? :\n1. Manually\n2. Randomly generated\n 👉 ").strip()
            
            #Input data manually
            if choice == "1":
                    while True:
                        try: 
                            n = int(input("Enter the number of buses for your system: ").strip())
                            break
                        except ValueError:
                            print("\nYOUR INPUT IS INVALID! \nPlease input an integer number!\n")
                    while True: 
                        try: 
                            choice = input("Choose how to provide the admittance matrix:\n1. Rectangular Form\n2. Polar Form\n👉").strip()
                            if choice == "1":
                                Y = get_admittance_matrix_rectangular(n)
                            elif choice == "2":
                                Y = get_admittance_matrix_polar(n)
                            else:
                                print("\nYOUR INPUT IS INVALID! \n Please input either 1 or 2\n")
                                continue
                            display_matrix(Y, "Admittance Matrix")
                            bus_data = get_bus_data(n)#calls bus data function take input
                            convergenceReached=newton_raphson_load_flow(bus_data, Y)
                            break
                            
                        except ValueError:
                            print("\nYOUR INPUT IS INVALID! \nPlease input an integer number!\n")
            
            #Develop data randomly
            elif choice == "2":
                while True:
                    try:
                        n = int(input("Enter the number of buses: ").strip())
                        Y = random_admittance_matrix(n)
                        display_matrix(Y, "Admittance Matrix (Randomly Generated)")
                        bus_data = random_bus_data(n)#calls random bus data function take input
                        convergenceReached=newton_raphson_load_flow(bus_data, Y)
                        break
                    except ValueError:
                        print("Please input an integer number!")
            else:
                print("\n\nYOUR INPUT IS INVALID! \nPlease try again\n")
        except ValueError:
            print("\n\nYOUR INPUT IS INVALID! \nPlease try again\n")
            continue
        if convergenceReached == True:
            break
            

    


    print("\nTHANK YOU. BE BACK SOON 😉\nNOTE:A FLAT VOLTAGE START IS ASSUMED FOR ALL INITIAL GUESSES")

if __name__ == "__main__":
    main()


    






    


