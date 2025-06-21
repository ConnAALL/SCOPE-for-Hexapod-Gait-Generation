"""
Hexapod robot (Mantis) controller. 
This script receives a candidate chromosome from the supervisor,
instantiates the new SCOPE controller and uses it to compute motor commands.
The output of the SCOPE controller (representing acceleration commands) is applied to update the motor positions.

This controller can also be used for testing the baseline controller.
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
    for i in range(6):          # For each leg
        leg_data = []
        for j in range(3):      # For each joint in the leg
            index = i * 3 + j   # Map to original motor index
            leg_data.extend(sensor_data[index])  # Append position, velocity, acceleration
        formatted_sensor_values.append(leg_data)
    return formatted_sensor_values

def main():
    mantis = Mantis()

    # Wait for candidate chromosome from the supervisor.
    candidate = mantis.wait_message()
    if isinstance(candidate, list) and "RESET" in candidate:  # If it is a chromosome and reset command, just end the controller. 
        return

    # Retrieve initial sensor data to form a sample for SCOPE initialization.
    initial_sensor_data = mantis.read_motor_positions()
    sensor_sample = restructure_sensor_data(initial_sensor_data)

    # Initialize the SCOPE controller with the candidate chromosome.
    controller = SCOPEController(chromosome=candidate,
                                 sensor_input_sample=np.array(sensor_sample))

    # Main control loop
    while mantis.simulation_step() != -1:
        
        # Read the sensor data and then process it for the controller.
        sensor_data = mantis.read_motor_positions()
        restructured_data = restructure_sensor_data(sensor_data)

        # Compute motor commands via the current SCOPE controller.
        scope_output = controller.forward(np.array(restructured_data))
        
        # Update motors using the specific configurations
        mantis.apply_sine(motor_values=scope_output)

        # Check for messages from the supervisor
        new_message = mantis.receive_message()
        if new_message:
            if isinstance(new_message, list) and "RESET" in new_message:
                # If the message is RESET, end the controller
                break
            elif isinstance(new_message, list) and "TRIAL_END" in new_message:
                # Send the stable count and reset the counter
                mantis.send_message(mantis.stableCount / mantis.trialSteps)
                mantis.stableCount = 0
                mantis.trialSteps = 0
            else:  # If a message is received and it is not a reset
                # Reinitialize the SCOPE controller with the new chromosome.
                controller.update_weights(new_message)
                

if __name__ == "__main__":
    main()
