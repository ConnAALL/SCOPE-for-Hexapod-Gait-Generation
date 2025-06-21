"""
Hexapod robot (Mantis) controller using the new SCOPE-based gait control.
This script receives a candidate chromosome from the supervisor,
instantiates the new SCOPE controller and uses it to compute motor commands.
The output of the SCOPE controller (representing acceleration commands) is applied
to update the motor positions.
"""

import os
import sys
import numpy as np
from RobotControl import Mantis

# Path to the SCOPE module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SCOPE import SCOPEController

def restructure_sensor_data(sensor_data: list):
    """
    Restructures sensor data from an 18x3 matrix (18 motors, each with [position, velocity, acceleration])
    into a 6x9 matrix. Each of the 6 rows represents a leg, and the 9 columns correspond to 3 joints
    """
    formatted_sensor_values = []
    for i in range(6):  # For each leg
        leg_data = []
        for j in range(3):  # Each leg has 3 joints
            index = i * 3 + j  # Map to original motor index
            leg_data.extend(sensor_data[index])
        formatted_sensor_values.append(leg_data)
    return formatted_sensor_values  # Returns a 6x9 matrix

def main():
    mantis = Mantis()

    # Wait for candidate chromosome from the supervisor.
    candidate = mantis.wait_message()
    if isinstance(candidate, list) and "RESET" in candidate:
        return

    # Retrieve initial sensor data to form a sample for SCOPE initialization.
    initial_sensor_data = mantis.read_motor_positions()             # Expected shape is 18x3
    sensor_sample = restructure_sensor_data(initial_sensor_data)    # Restructure to 6x9

    # Initialize the SCOPE controller with the candidate chromosome.
    controller = SCOPEController(chromosome=candidate, 
                                 sensor_input_sample=np.array(sensor_sample))

    while mantis.simulation_step() != -1:
        # Read the sensor data and then process it for the controller.
        sensor_data = mantis.read_motor_positions()
        restructured_data = restructure_sensor_data(sensor_data)

        # Compute motor commands via the current SCOPE controller.
        scope_output = controller.forward(np.array(restructured_data))
        
        # Update motors using the specific configurations
        mantis.update_motors(motor_values=scope_output,
                            sensor_data=sensor_data,
                            final_processing=None,
                            usage="sine")

        new_message = mantis.receive_message()
        if new_message:
            if isinstance(new_message, list) and "RESET" in new_message:
                # If the message is RESET, end the controller
                break
            elif isinstance(new_message, list) and "TRIAL_END" in new_message:
                mantis.send_message(mantis.stableCount)
            else:  # If a message is received and it is not a reset
                # Reinitialize the SCOPE controller with the new chromosome.
                controller.update_weights(new_message)
                mantis.stableCount = 0
        print(f"Step: {mantis.step_count} | Stable Count: {mantis.stableCount} | Ratio: {(mantis.stableCount / mantis.step_count) * 100}")
                

if __name__ == "__main__":
    main()
