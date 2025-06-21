"""
Main robot control file that handles the Mantis robot's motor control and communication with the supervisor.
There are two classes:
    1. Managing the robot itself
    2. Managing the individual motors (this class is used by the robot class)
"""

import os
import json
import math

from controller import Robot # type:ignore

config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.txt'))
with open(config_file, 'r') as file: config = json.load(file)

LIMIT_ROM_DEGREES = config['LIMIT_ROM_DEGREES'] # Maximum change per timestep (in radians)
SLICES = config['SLICES']                       # Number of slices for interpolation
F = config['F']                                 # Frequency for sine wave
GAIT_STEPS = config['GAIT_STEPS']               # Number of steps for the gait cycle
MOTOR_LIMITS_DEG = {                            # Movement limits for each motor in degrees
    name: tuple(bounds)
    for name, bounds in config['MOTOR_LIMITS_DEG'].items()
}

class Mantis:
    """
    Upper-level controller for the Mantis robot.
    Reads sensor data from motors and applies control commands computed by the SCOPE controller.
    """
    def __init__(self):
        self.init_robot()
        self.init_motors()
        self.init_communication()
        self.init_sensors()

    ######################### Robot Management ################################
    def init_robot(self):
        """Initialize the robot controller and set up basic parameters."""
        self.robot = Robot()  # Controller instance of the simulation
        self.time_step = int(self.robot.getBasicTimeStep())
        self.stableCount = 0
        self.trialSteps = 0
    
    def simulation_step(self):
        """Performs a basic simulation step"""
        return self.robot.step(self.time_step)
    
    ######################### Motor Management ################################
    def init_motors(self):
        """Initialize motor devices."""

        self.motor_names = ["RPC", "RPF", "RPT",
                            "RMC", "RMF", "RMT",
                            "RAC", "RAF", "RAT",
                            "LPC", "LPF", "LPT",
                            "LMC", "LMF", "LMT",
                            "LAC", "LAF", "LAT"]
        
        self.motors = [Motor(name, self.robot, self.time_step) for name in self.motor_names]
    
    def read_motor_positions(self):
        """
        Read sensor data (position, velocity, acceleration) from all motors.
        Returns a list of 18 lists (each inner list contains 3 values).
        """
        current_time = self.robot.getTime()
        sensor_values = [motor.update_sensor_data(current_time) for motor in self.motors]
        return sensor_values
    
    def set_all_motors(self, positions: list):
        """Set the target position for all motors."""
        for i, pos in enumerate(positions):
            self.motors[i].set_position(pos)
    
    ######################### Usage Types ##################################### 
    def apply_sine(self, motor_values: list):

        # Initialize last_positions if not already done (for 18 motors)
        if not hasattr(self, 'last_positions'):
            self.last_positions = [motor.sensor.getValue() for motor in self.motors]

        for _ in range(GAIT_STEPS):
            new_positions = []
            self.last_positions = [motor.sensor.getValue() for motor in self.motors]

            for i in range(len(self.motors)):
                # Extract parameters from input
                phase = motor_values[i * 3]
                amplitude = motor_values[i * 3 + 1]
                offset = motor_values[i * 3 + 2]

                currentTime = self.robot.getTime()
                # Compute target position from sine function
                raw_val = offset + amplitude * math.sin(2 * math.pi * F * currentTime + phase)

                # Limit rate of change
                LIMIT_ROM = math.radians(LIMIT_ROM_DEGREES)
                relative_min = self.last_positions[i] - LIMIT_ROM
                relative_max = self.last_positions[i] + LIMIT_ROM
                value = max(min(raw_val, relative_max), relative_min)

                # Get motor name and clamp to min/max (convert degrees to radians)
                motor_name = self.motors[i].motor.getName()
                deg_min, deg_max = MOTOR_LIMITS_DEG[motor_name]
                rad_min = math.radians(deg_min)
                rad_max = math.radians(deg_max)
                value = max(min(value, rad_max), rad_min)

                # Update list and track last position
                new_positions.append(value)

            # Update all motors with the new positions through multiple time steps
            self.move_motors_to_position(new_positions, slices=SLICES)

    def move_motors_to_position(self,
                                target_positions: list,
                                slices: int):
        start_positions = self.last_positions[:]  # Create a copy of the positions.
        # Perform interpolation in 'slices' steps.
        for slice_idx in range(1, slices + 1):
            new_positions = []
            for i in range(len(self.motors)):
                # Calculate the interpolated value for the current slice.
                interp_val = start_positions[i] + (target_positions[i] - start_positions[i]) * (slice_idx / slices)
                
                # Get motor-specific limits and clamp the interpolated value.
                motor_name = self.motors[i].motor.getName()
                deg_min, deg_max = MOTOR_LIMITS_DEG[motor_name]
                rad_min = math.radians(deg_min)
                rad_max = math.radians(deg_max)
                interp_val = max(min(interp_val, rad_max), rad_min)
                
                new_positions.append(interp_val)
            # Update all motors with the intermediate positions.
            self.set_all_motors(new_positions)
            # Update last_positions for the next cycle.
            self.last_positions = new_positions[:]

            # Check if the robot is stable and update the stable count accordingly.
            if self.is_currently_stable():
                self.stableCount += 1
                
            self.trialSteps += 1

            # Advance one simulation step for a smooth transition.
            self.simulation_step()

    ######################### Communication Functions #########################
    def init_communication(self):
        """Initialize communication with the supervisor."""
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.time_step)
    
    def send_message(self, message):
        self.emitter.send(json.dumps(message))
    
    def receive_message(self):
        if self.receiver.getQueueLength() > 0:
            msg = json.loads(self.receiver.getString())
            self.receiver.nextPacket()
            return msg
        return None
    
    def wait_message(self):
        """Wait until a message is received from the supervisor."""
        msg = self.receive_message()
        while msg is None:
            msg = self.receive_message()
            self.simulation_step()
        return msg
    
    ######################### Sensor Management ###############################
    def init_sensors(self):
        """
        Initialize GPS and touch sensors for the robot's leg tips.
        
        This method retrieves the sensors by name from the robot, enables them,
        and stores them in dictionaries for later use.
        """
        # Define sensor names for each type.
        gps_names = ["rps_gps", "rms_gps", "ras_gps", "lps_gps", "lms_gps", "las_gps"]
        touch_names = ["rps_touch", "rms_touch", "ras_touch", "lps_touch", "lms_touch", "las_touch"]
        
        # Initialize dictionaries to store sensor devices.
        self.gps_sensors = {}
        self.touch_sensors = {}
        
        # Enable each GPS sensor.
        for name in gps_names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.time_step)
            self.gps_sensors[name] = sensor
        
        # Enable each touch sensor.
        for name in touch_names:
            sensor = self.robot.getDevice(name)
            sensor.enable(self.time_step)
            self.touch_sensors[name] = sensor

        # GPS sensor for the body of the robot
        self.body_gps = self.robot.getDevice("gps_body")
        self.body_gps.enable(self.time_step)

    ######################### Checking Static Stability #######################
    def get_body_position(self):
        """
        Returns the robot's body position as a list [x, y, z].
        """
        return self.body_gps.getValues()

    def _get_feet_touching_ground(self):
        """
        Return a list of foot positions for which the touch sensor value is 1.
        Each position is returned as a 3-tuple (x, y, z). It uses the corresponding GPS sensor.
        """
        feet_positions = []
        for touch_name, touch_sensor in self.touch_sensors.items():
            if touch_sensor.getValue() > 0:  # As far as I understood, touch sensor works as 1/0 but still I said larger than 0
                gps_key = touch_name.replace("_touch", "_gps")  # It is the same begginnign with a different tag
                foot_pos = self.gps_sensors[gps_key].getValues()  # If the foot is touching the ground, get its GPS position
                feet_positions.append((foot_pos[0], foot_pos[1], foot_pos[2]))
        return feet_positions

    def _convert_to_2d(self, com, leg_points):
        """
        Convert 3D points to 2D for stability checking.
        Only keep leg points whose z value is below the z-value of the center of mass.
        Returns a tuple: (com_2d, list_of_2d_leg_points).
        """
        com_x, com_y, com_z = com
        filtered_points = [pt for pt in leg_points if pt[2] < com_z]
        pts_2d = [(x, y) for (x, y, _) in filtered_points]
        return (com_x, com_y), pts_2d

    def _convex_hull(self, points):
        """
        Compute the convex hull of a set of 2D points using Andrew's monotone chain algorithm.
        Returns a list of vertices that make up the convex hull.
        """
        points = sorted(set(points))
        if len(points) <= 1:
            return points

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        # Concatenate lower and upper hulls; omit the last point of each because it's repeated.
        return lower[:-1] + upper[:-1]

    def _is_statically_stable(self, leg_points, com):
        """
        Determine if the robot's center of mass (com) is inside the support polygon.
        The support polygon is defined as the convex hull of the leg points (projected to 2D).
        """
        com_2d, pts_2d = self._convert_to_2d(com, leg_points)
        polygon_edges = self._convex_hull(pts_2d)
        if len(polygon_edges) < 3:
            return False  # Not enough points to form a polygon.
        com_x, com_y = com_2d
        counter = 0
        n = len(polygon_edges)
        for i in range(n):
            x1, y1 = polygon_edges[i]
            x2, y2 = polygon_edges[(i + 1) % n]

            # Count intersections with a ray from the COM.
            if (com_y < y1) != (com_y < y2):
                intersect_x = x1 + ((com_y - y1) / (y2 - y1)) * (x2 - x1)
                if com_x < intersect_x:
                    counter += 1
        return (counter % 2) == 1

    def is_currently_stable(self):
        """
        Checks whether the robot is currently statically stable.
        It collects the positions of feet in contact (touch sensor > 0),
        retrieves the robot's body position (COM), and determines if the COM is inside
        the convex hull of the feet in contact.
        Returns True if stable, False otherwise.
        """
        feet_positions = self._get_feet_touching_ground()
        com = self.get_body_position()
        return self._is_statically_stable(feet_positions, com)

class Motor:
    """
    Controls an individual motor.
    Manages position sensors and updates motor positions.
    """
    def __init__(self, motor_name: str, robot: Robot, time_step: int):
        self.robot = robot
        self.motor = robot.getDevice(motor_name)
        self.time_step = time_step
        self.init_position_sensor()
        self.init_value_tracking()
    
    ######################### Setup Functions #################################
    def init_value_tracking(self):
        """Initialize variables for tracking sensor values."""
        self.prev_sensor = None
        self.prev_velocity = 0.0
        self.prev_acceleration = 0.0
        self.prev_time = None

    def init_position_sensor(self):
        """Initialize the position sensor for the motor."""
        self.sensor = self.motor.getPositionSensor()
        try:
            self.sensor.enable(self.time_step)
        except Exception as e:
            print(f"Error enabling sensor for motor {self.motor.getName()}: {e}")
    
    ######################### Motor Control ###################################
    def set_position(self, pos: float):
        self.motor.setPosition(pos)
    
    def update_sensor_data(self, current_time: float) -> list:
        """ Update the sensor data for the motor."""
        sensor_val = self.sensor.getValue()
        position_deg = sensor_val
        if self.prev_time is None:
            dt = self.time_step / 1000.0
            velocity = 0.0
            acceleration = 0.0
        else:
            dt = current_time - self.prev_time
            if dt <= 0:
                dt = self.time_step / 1000.0
            velocity = ((sensor_val - self.prev_sensor) / dt) if self.prev_sensor is not None else 0.0
            acceleration = (velocity - self.prev_velocity) / dt
        self.prev_sensor = sensor_val
        self.prev_velocity = velocity
        self.prev_acceleration = acceleration
        self.prev_time = current_time
        return [position_deg, velocity, acceleration]