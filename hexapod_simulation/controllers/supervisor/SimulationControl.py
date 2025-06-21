"""
Simulation control file that has the main supervisor controller of the simualation. 
It hosts the SimulationControl class that is responsible for setting up the simulation, 
communication and logging functionalities.
"""

import os
import time
import json
import math
import csv

from controller import Supervisor # type:ignore

class SimulationControl:
    """
    Class for controlling the simulation environment.
    It manages simulation resets,
    communication with the robot,
    and evaluation of candidate controllers.
    """
    def __init__(self, startingPosition = [0, 0, 0]):
        self.STARTING_POSITION = startingPosition  # The starting position is 0, 0, 0 by default
        self.init_supervisor()
        self.init_mantis()
        self.reset_simulation()
        self.init_logging()

    ######################### Supervisor Management ###########################
    def init_supervisor(self):
        """Initialize the supervisor and related functionalities."""
        self.supervisor = Supervisor()
        self.root = self.supervisor.getRoot()
        self.children_field = self.root.getField("children")
        self.def_name = "mantis"

        with open("mantis_robot.txt", "r") as f:
            # There should be a mantis_robtot.txt file in the same directort as this script.
            # It stores the mantis robot definition in the Webots PROTO format.
            self.mantis_node_str = f.read()

        self.time_step = int(self.supervisor.getBasicTimeStep())
        self.init_communication()

    def simulation_step(self):
        """Performs a basic simulation step"""
        return self.supervisor.step(self.time_step)

    ######################### Robot Management ################################
    def init_mantis(self):
        """Initialize the hexapod robot."""
        self.mantis_node = self.supervisor.getFromDef(self.def_name)
        self.position_field = self.mantis_node.getField("translation")
        self.rotation_field = self.mantis_node.getField("rotation")

    def remove_mantis(self):
        """Remove the hexapod robot from the simulation."""
        self.mantis_node.remove()

    def add_mantis(self):
        """Add a new hexapod robot to the simulation."""
        self.children_field.importMFNodeFromString(-1, self.mantis_node_str)
        self.init_mantis()

    def reset_simulation(self):
        """
        Reset the simulation by stopping the current controller,
        removing the robot, adding a new one, and resetting its position.
        """
        self.stop_mantis_controller()
        self.remove_mantis()
        self.add_mantis()
        self.set_position(self.STARTING_POSITION)
        self.set_rotation([0, 0, 1, 0])
        self.previous_x, self.previous_y = self.STARTING_POSITION[0], self.STARTING_POSITION[1]
        self.supervisor.simulationResetPhysics()

    ######################### Communication Functions #########################
    def init_communication(self):
        """Initialize the emitter and receiver devices for communication"""
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)
    
    def send_message(self, message):
        """Send a message (candidate chromosome) to the robot."""
        self.emitter.send(json.dumps(message))
    
    def receive_message(self):
        """Receive a message from the robot."""
        if self.receiver.getQueueLength() > 0:
            msg = json.loads(self.receiver.getString())
            self.receiver.nextPacket()
            return msg
        return None
    
    def wait_message(self):
        """Wait until a message is received from the robot."""
        msg = self.receive_message()
        while msg is None:
            msg = self.receive_message()
            self.simulation_step()
        return msg

    def stop_mantis_controller(self):
        """Send a stop command to the robot."""
        self.send_message(["RESET"])
        for _ in range(3):
            self.simulation_step()

    ######################### Location Functions ##############################
    def set_position(self, new_position):
        """Set the position of the hexapod robot."""
        self.position_field.setSFVec3f(new_position)
    
    def set_rotation(self, new_rotation):
        """Set the rotation of the hexapod robot."""
        self.rotation_field.setSFRotation(new_rotation)
    
    def get_position(self):
        """Get the current location of the hexapod robot."""
        return self.position_field.getSFVec3f()
    
    def get_position_change(self):
        """Compute the displacement from the previous position."""
        current_x, current_y, _ = self.get_position()
        dx = current_x - self.previous_x
        dy = current_y - self.previous_y
        self.previous_x, self.previous_y = current_x, current_y
        return math.sqrt(dx ** 2 + dy ** 2)

    ######################### Data Logging Functions ###########################
    def init_logging(self):
        """Initialize the logging system by creating necessary directories and a CSV file."""
        # Define the base data directory (two directories up)
        base_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "DATA"))
        os.makedirs(base_dir, exist_ok=True)

        # Create a subdirectory with a timestamp
        self.timestamp = time.strftime("%y%m%d_%H%M%S")
        self.log_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.log_dir)

        # Create a directory for storing best individuals' weights
        os.makedirs(os.path.join(self.log_dir, "BEST_INDV"))

        # Create CSV file for logging fitness scores
        runName = self.get_run_name()
        self.log_file = os.path.join(self.log_dir, f"{runName}_{self.timestamp}.csv")
        with open(self.log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Generation",
                "Individual",
                "DistanceDifference",
                "StabilityRatio",
                "Fitness",
                ])

    def log_fitness(self,
                    generation,
                    solution,
                    fitness,
                    distance_difference=None,
                    stability_ratio=None):
        """Log generation, individual, distance, stability ratio, and fitness."""
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                generation,
                solution,
                distance_difference if distance_difference is not None else "",
                stability_ratio    if stability_ratio    is not None else "",
                fitness
                ])

    def log_parameters(self, parameters: dict):
        """
        Log the given parameters dictionary into a text file.
        If a value is a dictionary, its key-value pairs are logged recursively.
        """
        params_file = os.path.join(self.log_dir, f"{self.timestamp}_parameters.txt")
        with open(params_file, mode="w") as file:
            def write_params(params, indent=0):
                for key, value in params.items():
                    if isinstance(value, dict):
                        file.write("    " * indent + f"{key}:\n")
                        write_params(value, indent + 1)
                    else:
                        file.write("    " * indent + f"{key} - {value}\n")
            write_params(parameters)
        print("Parameters logged!")
    
    def get_run_name(self):
        """
        Get the run name based on the configurations
        """
        config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.txt'))
        with open(config_file, 'r') as file: config = json.load(file)
        
        MODEL_TYPE = config["MODEL_TYPE"]
        FITNESS = config["STABLE_FITNESS"]

        if MODEL_TYPE == "scope":
            run_name = f"scope"
        elif MODEL_TYPE == "ssga":
            run_name = f"ssga"
        else: 
            raise ValueError(f"Unknown model type: {MODEL_TYPE}")
        
        if FITNESS == 1:
            run_name += "_PF"
        elif FITNESS == 0:
            run_name += "_DF"
        else:
            raise ValueError(f"Unknown fitness type: {FITNESS}")
        
        return run_name