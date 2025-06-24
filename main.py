import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import random
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Array
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Point:
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Road:
    """Represents a road segment"""
    start: Point
    end: Point
    id: str
    max_speed: float = 50.0
    lanes: int = 2
    
    def get_length(self) -> float:
        return self.start.distance_to(self.end)
    
    def get_point_at_progress(self, progress: float) -> Point:
        """Get point along road at given progress (0.0 to 1.0)"""
        progress = max(0.0, min(1.0, progress))
        x = self.start.x + (self.end.x - self.start.x) * progress
        y = self.start.y + (self.end.y - self.start.y) * progress
        return Point(x, y)

@dataclass
class Vehicle:
    """Represents a vehicle with realistic movement"""
    id: int
    position: Point
    speed: float
    max_speed: float
    current_road: Optional[Road]
    road_progress: float  # 0.0 to 1.0 along current road
    route: List[Road]
    route_index: int
    zone_origin: str
    zone_destination: str
    wait_time: float = 0.0
    total_travel_time: float = 0.0
    color: str = "blue"
    
    def update_position(self, time_step: float, traffic_factor: float = 1.0):
        """Update vehicle position along route"""
        if not self.current_road or not self.route:
            return
            
        # Calculate actual speed based on traffic conditions
        actual_speed = min(self.speed * traffic_factor, self.max_speed)
        
        # Convert speed from km/h to pixels per second (adjusted for larger network)
        speed_pixels_per_sec = actual_speed * 0.6  # Increased from 0.4 to 0.6 for better movement on larger roads
        
        # Calculate distance traveled in this time step
        distance_traveled = speed_pixels_per_sec * time_step
        
        # Update position along current road
        road_length = self.current_road.get_length()
        if road_length > 0:
            progress_increment = distance_traveled / road_length
            self.road_progress += progress_increment
            
            # Update total travel time
            self.total_travel_time += time_step
            
            # Add wait time if traffic is slow
            if traffic_factor < 0.5:
                self.wait_time += time_step
            
            # Check if we've completed current road
            if self.road_progress >= 1.0:
                self.road_progress = 0.0
                self.route_index += 1
                
                # Check if route is complete
                if self.route_index >= len(self.route):
                    return "completed"
                else:
                    self.current_road = self.route[self.route_index]
            
            # Update actual position
            self.position = self.current_road.get_point_at_progress(self.road_progress)
        
        return "active"

class TrafficLight:
    """Traffic light with realistic timing"""
    def __init__(self, position: Point, light_id: str, green_duration: float = 25.0, red_duration: float = 15.0):
        self.position = position
        self.id = light_id
        self.green_duration = green_duration
        self.red_duration = red_duration
        self.current_state = "GREEN"
        self.time_in_state = 0.0
        self.affected_roads: List[Road] = []
    
    def update(self, time_step: float):
        """Update traffic light state"""
        self.time_in_state += time_step
        
        if self.current_state == "GREEN" and self.time_in_state >= self.green_duration:
            self.current_state = "RED"
            self.time_in_state = 0.0
        elif self.current_state == "RED" and self.time_in_state >= self.red_duration:
            self.current_state = "GREEN"
            self.time_in_state = 0.0
    
    def get_traffic_factor(self, road: Road) -> float:
        """Get traffic factor for vehicles on affected roads"""
        if road in self.affected_roads:
            return 1.0 if self.current_state == "GREEN" else 0.1
        return 1.0

class RoadNetwork:
    """Manages the road network and routing"""
    def __init__(self):
        self.roads: Dict[str, Road] = {}
        self.intersections: Dict[str, Point] = {}
        self.traffic_lights: Dict[str, TrafficLight] = {}
        # Properly centered and sized for 600x460 canvas
        self.zone_centers = {
            'North': Point(300, 100),    # Top center
            'South': Point(300, 360),    # Bottom center  
            'East': Point(500, 230),     # Right center
            'West': Point(100, 230)      # Left center
        }
        self.build_network()
    
    def build_network(self):
        """Build a realistic road network"""
        # Define major intersections - larger and properly centered
        self.intersections = {
            'center': Point(300, 230),       # Canvas center
            'north_mid': Point(300, 165),    # Between center and north zone
            'south_mid': Point(300, 295),    # Between center and south zone
            'east_mid': Point(400, 230),     # Between center and east zone
            'west_mid': Point(200, 230),     # Between center and west zone
            'north_zone': Point(300, 100),   # North zone position
            'south_zone': Point(300, 360),   # South zone position
            'east_zone': Point(500, 230),    # East zone position
            'west_zone': Point(100, 230)     # West zone position
        }
        
        # Create main roads connecting zones
        road_configs = [
            # North-South main road
            ('north_main_1', 'north_zone', 'north_mid'),
            ('north_main_2', 'north_mid', 'center'),
            ('south_main_1', 'center', 'south_mid'),
            ('south_main_2', 'south_mid', 'south_zone'),
            
            # East-West main road
            ('west_main_1', 'west_zone', 'west_mid'),
            ('west_main_2', 'west_mid', 'center'),
            ('east_main_1', 'center', 'east_mid'),
            ('east_main_2', 'east_mid', 'east_zone'),
            
            # Additional connecting roads
            ('north_east_1', 'north_mid', 'east_mid'),
            ('north_west_1', 'north_mid', 'west_mid'),
            ('south_east_1', 'south_mid', 'east_mid'),
            ('south_west_1', 'south_mid', 'west_mid'),
        ]
        
        # Create road objects
        for road_id, start_name, end_name in road_configs:
            start_point = self.intersections[start_name]
            end_point = self.intersections[end_name]
            self.roads[road_id] = Road(start_point, end_point, road_id, max_speed=random.uniform(40, 60))
        
        # Create traffic lights at major intersections
        light_positions = ['center', 'north_mid', 'south_mid', 'east_mid', 'west_mid']
        for i, pos_name in enumerate(light_positions):
            position = self.intersections[pos_name]
            light = TrafficLight(position, f"light_{i}", 
                               green_duration=random.uniform(20, 30),
                               red_duration=random.uniform(10, 20))
            
            # Assign roads affected by this light - appropriate radius for larger network
            light.affected_roads = [road for road in self.roads.values() 
                                  if (road.start.distance_to(position) < 45 or   # Increased from 35 to 45
                                      road.end.distance_to(position) < 45)]
            
            self.traffic_lights[f"light_{i}"] = light
    
    def find_route(self, origin_zone: str, destination_zone: str) -> List[Road]:
        """Find route between zones using simple pathfinding"""
        if origin_zone == destination_zone:
            return []
        
        # Define route patterns (simplified for demonstration)
        route_patterns = {
            ('North', 'South'): ['north_main_1', 'north_main_2', 'south_main_1', 'south_main_2'],
            ('South', 'North'): ['south_main_2', 'south_main_1', 'north_main_2', 'north_main_1'],
            ('West', 'East'): ['west_main_1', 'west_main_2', 'east_main_1', 'east_main_2'],
            ('East', 'West'): ['east_main_2', 'east_main_1', 'west_main_2', 'west_main_1'],
            ('North', 'East'): ['north_main_1', 'north_main_2', 'east_main_1', 'east_main_2'],
            ('North', 'West'): ['north_main_1', 'north_main_2', 'west_main_2', 'west_main_1'],
            ('South', 'East'): ['south_main_2', 'south_main_1', 'east_main_1', 'east_main_2'],
            ('South', 'West'): ['south_main_2', 'south_main_1', 'west_main_2', 'west_main_1'],
            ('East', 'North'): ['east_main_2', 'east_main_1', 'north_main_2', 'north_main_1'],
            ('East', 'South'): ['east_main_2', 'east_main_1', 'south_main_1', 'south_main_2'],
            ('West', 'North'): ['west_main_1', 'west_main_2', 'north_main_2', 'north_main_1'],
            ('West', 'South'): ['west_main_1', 'west_main_2', 'south_main_1', 'south_main_2'],
        }
        
        route_ids = route_patterns.get((origin_zone, destination_zone), [])
        return [self.roads[road_id] for road_id in route_ids if road_id in self.roads]

class TrafficSimulation:
    """Main simulation engine"""
    def __init__(self):
        self.road_network = RoadNetwork()
        self.vehicles: Dict[int, Vehicle] = {}
        self.next_vehicle_id = 1
        self.simulation_time = 0.0
        self.total_vehicles_generated = 0
        self.total_vehicles_completed = 0
        
        # Performance metrics
        self.zone_metrics = {
            'North': {'vehicles': 0, 'congestion': 0.0, 'avg_speed': 0.0},
            'South': {'vehicles': 0, 'congestion': 0.0, 'avg_speed': 0.0},
            'East': {'vehicles': 0, 'congestion': 0.0, 'avg_speed': 0.0},
            'West': {'vehicles': 0, 'congestion': 0.0, 'avg_speed': 0.0}
        }
    
    def generate_vehicle(self) -> Optional[Vehicle]:
        """Generate a new vehicle with random origin and destination"""
        zones = list(self.road_network.zone_centers.keys())
        origin = random.choice(zones)
        destination = random.choice([z for z in zones if z != origin])
        
        route = self.road_network.find_route(origin, destination)
        if not route:
            return None
        
        # Create vehicle at start of first road
        start_road = route[0]
        vehicle = Vehicle(
            id=self.next_vehicle_id,
            position=start_road.start,
            speed=random.uniform(30, 60),
            max_speed=random.uniform(50, 80),
            current_road=start_road,
            road_progress=0.0,
            route=route,
            route_index=0,
            zone_origin=origin,
            zone_destination=destination,
            color=random.choice(['blue', 'red', 'green', 'orange', 'purple'])
        )
        
        self.next_vehicle_id += 1
        self.total_vehicles_generated += 1
        return vehicle
    
    def update_traffic_lights(self, time_step: float):
        """Update all traffic lights"""
        for light in self.road_network.traffic_lights.values():
            light.update(time_step)
    
    def get_traffic_factor_for_vehicle(self, vehicle: Vehicle) -> float:
        """Calculate traffic factor for a vehicle based on lights and congestion"""
        base_factor = 1.0
        
        # Check traffic lights
        for light in self.road_network.traffic_lights.values():
            light_factor = light.get_traffic_factor(vehicle.current_road)
            if light_factor < base_factor:
                base_factor = light_factor
        
        # Add congestion factor based on nearby vehicles
        nearby_vehicles = 0
        for other_vehicle in self.vehicles.values():
            if (other_vehicle.id != vehicle.id and 
                other_vehicle.current_road == vehicle.current_road):
                nearby_vehicles += 1
        
        # Reduce speed based on congestion
        congestion_factor = max(0.3, 1.0 - nearby_vehicles * 0.1)
        
        return base_factor * congestion_factor
    
    def simulate_step(self, time_step: float):
        """Simulate one time step"""
        self.simulation_time += time_step
        
        # Update traffic lights
        self.update_traffic_lights(time_step)
        
        # Update vehicles
        vehicles_to_remove = []
        for vehicle in list(self.vehicles.values()):
            traffic_factor = self.get_traffic_factor_for_vehicle(vehicle)
            status = vehicle.update_position(time_step, traffic_factor)
            
            if status == "completed":
                vehicles_to_remove.append(vehicle.id)
                self.total_vehicles_completed += 1
        
        # Remove completed vehicles
        for vehicle_id in vehicles_to_remove:
            del self.vehicles[vehicle_id]
        
        # Generate new vehicles occasionally
        if random.random() < 0.1 and len(self.vehicles) < 50:  # Limit total vehicles
            new_vehicle = self.generate_vehicle()
            if new_vehicle:
                self.vehicles[new_vehicle.id] = new_vehicle
        
        # Update zone metrics
        self.update_zone_metrics()
    
    def update_zone_metrics(self):
        """Update metrics for each zone"""
        # Reset counters
        for zone in self.zone_metrics:
            self.zone_metrics[zone] = {'vehicles': 0, 'congestion': 0.0, 'avg_speed': 0.0}
        
        # Count vehicles in each zone area
        for vehicle in self.vehicles.values():
            # Determine which zone the vehicle is currently in
            min_distance = float('inf')
            closest_zone = None
            
            for zone_name, zone_center in self.road_network.zone_centers.items():
                distance = vehicle.position.distance_to(zone_center)
                if distance < min_distance:
                    min_distance = distance
                    closest_zone = zone_name
            
            if closest_zone and min_distance < 120:  # Increased for larger zones (80px radius + buffer)
                self.zone_metrics[closest_zone]['vehicles'] += 1
                self.zone_metrics[closest_zone]['avg_speed'] += vehicle.speed
        
        # Calculate averages and congestion
        for zone_name in self.zone_metrics:
            metrics = self.zone_metrics[zone_name]
            if metrics['vehicles'] > 0:
                metrics['avg_speed'] /= metrics['vehicles']
                metrics['congestion'] = min(1.0, metrics['vehicles'] / 20.0)  # Max 20 vehicles per zone
            else:
                metrics['avg_speed'] = 0.0
                metrics['congestion'] = 0.0
    
    def get_state(self) -> Dict:
        """Get current simulation state for GUI"""
        return {
            'vehicles': {v_id: {
                'position': {'x': v.position.x, 'y': v.position.y},
                'color': v.color,
                'speed': v.speed,
                'origin': v.zone_origin,
                'destination': v.zone_destination
            } for v_id, v in self.vehicles.items()},
            'traffic_lights': {l_id: {
                'position': {'x': l.position.x, 'y': l.position.y},
                'state': l.current_state
            } for l_id, l in self.road_network.traffic_lights.items()},
            'roads': {r_id: {
                'start': {'x': r.start.x, 'y': r.start.y},
                'end': {'x': r.end.x, 'y': r.end.y}
            } for r_id, r in self.road_network.roads.items()},
            'zone_metrics': self.zone_metrics,
            'total_vehicles': len(self.vehicles),
            'total_generated': self.total_vehicles_generated,
            'total_completed': self.total_vehicles_completed,
            'simulation_time': self.simulation_time
        }

def parallel_simulation_worker(process_id: int, result_queue: Queue, control_value: Value, time_step: float = 0.1):
    """Simplified worker function for parallel simulation"""
    
    try:
        # Create independent simulation for this process
        simulation = TrafficSimulation()
        
        # Add some randomization per process to make them different
        random.seed(process_id * 42 + int(time.time()))  # Different seed per process
        
        # Slightly different vehicle generation rates per process
        generation_rate = 0.1 + (process_id * 0.02)  # 0.1, 0.12, 0.14, 0.16
        
        simulation_time = 0.0
        step_count = 0
        
        print(f"Process {process_id} started successfully")
        
        while control_value.value == 1:
            start_time = time.time()
            
            # Run simulation step
            simulation.simulate_step(time_step)
            simulation_time += time_step
            step_count += 1
            
            # Modify vehicle generation rate slightly for this process
            if random.random() < generation_rate and len(simulation.vehicles) < 50:
                new_vehicle = simulation.generate_vehicle()
                if new_vehicle:
                    simulation.vehicles[new_vehicle.id] = new_vehicle
            
            # Get state and add process information
            state = simulation.get_state()
            state['process_id'] = process_id
            state['simulation_time'] = simulation_time
            state['step_count'] = step_count
            
            # Send state to main process
            try:
                # Clear old states to prevent queue buildup
                while not result_queue.empty():
                    try:
                        result_queue.get_nowait()
                    except:
                        break
                result_queue.put(state, block=False)
            except Exception as e:
                print(f"Process {process_id} queue error: {e}")
                pass  # Queue might be full
            
            # Control simulation speed
            elapsed = time.time() - start_time
            sleep_time = max(0, time_step - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print(f"Process {process_id} stopped after {step_count} steps")
    
    except Exception as e:
        print(f"Process {process_id} error: {e}")
        import traceback
        traceback.print_exc()

def simulation_worker(result_queue: Queue, control_value: Value, time_step: float = 0.1):
    """Worker function for simulation process"""
    simulation = TrafficSimulation()
    
    while control_value.value == 1:
        start_time = time.time()
        
        simulation.simulate_step(time_step)
        
        # Send state to main process
        state = simulation.get_state()
        try:
            # Clear old states and send new one
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except:
                    break
            result_queue.put(state, block=False)
        except:
            pass  # Queue might be full
        
        # Control simulation speed
        elapsed = time.time() - start_time
        sleep_time = max(0, time_step - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

class TrafficSimulationGUI:
    """Enhanced GUI with proper road network visualization"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Smart City Traffic Flow Simulation - Real-time Road Network")
        self.root.geometry("1200x800")  # Reduced from 1600x1000
        
        # Simulation state
        self.is_running = False
        self.simulation_mode = "serial"
        self.current_state = None
        
        # Serial simulation objects
        self.serial_simulation = None
        self.serial_thread = None
        
        # Parallel simulation objects
        self.parallel_processes = []
        self.result_queues = []
        self.control_values = []
        
        # Performance tracking
        self.performance_data = {
            'serial': {'execution_times': [], 'avg_processing_time': 0, 'total_steps': 0},
            'parallel': {'execution_times': [], 'avg_processing_time': 0, 'total_steps': 0}
        }
        
        # GUI update tracking
        self.last_update_time = time.time()
        self.frame_count = 0
        
        # Create GUI
        self.create_widgets()
        
        # Start update loop
        self.update_gui()
    
    def create_widgets(self):
        """Create GUI widgets"""
        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(control_frame, text="Processing Mode")
        mode_frame.pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value="serial")
        ttk.Radiobutton(mode_frame, text="Serial Processing", variable=self.mode_var, 
                       value="serial").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(mode_frame, text="Parallel Processing", variable=self.mode_var, 
                       value="parallel").pack(anchor=tk.W, padx=5, pady=2)
        
        # Simulation controls
        controls = ttk.LabelFrame(control_frame, text="Simulation Controls")
        controls.pack(side=tk.LEFT, padx=5)
        
        self.start_button = ttk.Button(controls, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_button = ttk.Button(controls, text="Stop Simulation", command=self.stop_simulation, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.reset_button = ttk.Button(controls, text="Reset", command=self.reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.compare_button = ttk.Button(controls, text="Performance Comparison", command=self.show_comparison_window)
        self.compare_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Parameters
        params = ttk.LabelFrame(control_frame, text="Parameters")
        params.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(params, text="Speed:").grid(row=0, column=0, padx=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        ttk.Scale(params, from_=0.1, to=3.0, variable=self.speed_var, orient="horizontal", length=100).grid(row=0, column=1, padx=5)
        
        # Status
        status = ttk.LabelFrame(control_frame, text="Status")
        status.pack(side=tk.RIGHT, padx=5)
        
        self.status_label = ttk.Label(status, text="Ready")
        self.status_label.pack(padx=10, pady=5)
        
        # Main content
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left side - Road network visualization
        sim_frame = ttk.LabelFrame(main_frame, text="Road Network & Traffic Flow")
        sim_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Canvas for road network - reduced size
        self.canvas = tk.Canvas(sim_frame, bg="lightgray", width=600, height=460)  # Reduced from 800x600
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right side - Information panels
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Create notebook for different info views
        self.info_notebook = ttk.Notebook(info_frame)
        self.info_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Metrics tab
        metrics_tab = ttk.Frame(self.info_notebook)
        self.info_notebook.add(metrics_tab, text="Zone Metrics")
        
        self.metrics_text = scrolledtext.ScrolledText(metrics_tab, width=30, height=12)  # Reduced from 35x15
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Vehicle info tab
        vehicle_tab = ttk.Frame(self.info_notebook)
        self.info_notebook.add(vehicle_tab, text="Vehicle Info")
        
        self.vehicle_text = scrolledtext.ScrolledText(vehicle_tab, width=30, height=12)  # Reduced from 35x15
        self.vehicle_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Performance tab
        perf_tab = ttk.Frame(self.info_notebook)
        self.info_notebook.add(perf_tab, text="Performance")
        
        self.perf_text = tk.Text(perf_tab, width=30, height=12)  # Reduced from 35x15
        self.perf_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Comparison tab
        comparison_tab = ttk.Frame(self.info_notebook)
        self.info_notebook.add(comparison_tab, text="Comparison")
        
        self.comparison_text = scrolledtext.ScrolledText(comparison_tab, width=30, height=12)  # Reduced from 35x15
        self.comparison_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize display
        self.draw_road_network()
    
    def draw_road_network(self):
        """Draw the road network on canvas"""
        self.canvas.delete("all")
        
        # Create a sample road network for display
        road_network = RoadNetwork()
        
        # Draw a subtle grid to help with positioning (optional - can be removed)
        # for i in range(0, 600, 100):
        #     self.canvas.create_line(i, 0, i, 460, fill="lightgray", dash=(1, 3), tags="grid")
        # for i in range(0, 460, 100):
        #     self.canvas.create_line(0, i, 600, i, fill="lightgray", dash=(1, 3), tags="grid")
        
        # Draw roads - thicker for better visibility
        for road in road_network.roads.values():
            self.canvas.create_line(
                road.start.x, road.start.y, road.end.x, road.end.y,
                fill="darkgray", width=8, tags="road"  # Increased from 6 to 8
            )
            # Draw lane dividers
            self.canvas.create_line(
                road.start.x, road.start.y, road.end.x, road.end.y,
                fill="yellow", width=2, dash=(6, 4), tags="lane_divider"  # Thicker dividers
            )
        
        # Draw intersections - larger circles
        for name, point in road_network.intersections.items():
            self.canvas.create_oval(
                point.x - 12, point.y - 12, point.x + 12, point.y + 12,  # Increased from 10 to 12
                fill="lightblue", outline="blue", width=2, tags="intersection"  # Thicker border
            )
        
        # Draw zone areas - larger and properly positioned
        for zone_name, center in road_network.zone_centers.items():
            self.canvas.create_oval(
                center.x - 80, center.y - 80, center.x + 80, center.y + 80,  # Increased from 60 to 80
                fill="", outline="green", width=3, dash=(8, 4), tags=f"zone_{zone_name}"  # Thicker border
            )
            # Position labels properly for each zone
            if zone_name == "North":
                label_y = center.y + 95  # Below the zone
            elif zone_name == "South":
                label_y = center.y - 95  # Above the zone
            elif zone_name == "East":
                label_x, label_y = center.x - 95, center.y  # Left of zone
            else:  # West
                label_x, label_y = center.x + 95, center.y  # Right of zone
            
            # Adjust x position for East/West zones
            if zone_name in ["East", "West"]:
                self.canvas.create_text(
                    label_x, label_y, text=f"{zone_name}\nZone",
                    font=("Arial", 10, "bold"), fill="green", tags=f"zone_label_{zone_name}"
                )
            else:
                self.canvas.create_text(
                    center.x, label_y, text=f"{zone_name} Zone",
                    font=("Arial", 10, "bold"), fill="green", tags=f"zone_label_{zone_name}"
                )
        
        # Draw traffic lights - larger size
        for light in road_network.traffic_lights.values():
            self.canvas.create_rectangle(
                light.position.x - 8, light.position.y - 8,  # Increased from 6 to 8
                light.position.x + 8, light.position.y + 8,
                fill="red", outline="black", width=2, tags=f"traffic_light_{light.id}"  # Thicker border
            )
    
    def start_simulation(self):
        """Start the simulation"""
        if self.is_running:
            return
        
        self.is_running = True
        self.simulation_mode = self.mode_var.get()
        
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_label.config(text=f"Running ({self.simulation_mode.title()})...")
        
        if self.simulation_mode == "serial":
            self.start_serial_simulation()
        else:
            self.start_parallel_simulation()
        
        self.log_message(f"Started {self.simulation_mode} simulation with real-time traffic flow")
    
    def start_serial_simulation(self):
        """Start serial simulation - all zones in one process"""
        self.serial_simulation = TrafficSimulation()
        
        # Start simulation in a separate thread (but single process)
        self.serial_thread = threading.Thread(target=self.run_serial_simulation)
        self.serial_thread.daemon = True
        self.serial_thread.start()
    
    def start_parallel_simulation(self):
        """Start parallel simulation - simplified approach"""
        # Create 4 parallel simulation processes (each runs full simulation)
        num_processes = 4
        
        # Clear previous processes
        self.parallel_processes = []
        self.result_queues = []
        self.control_values = []
        
        print(f"Starting {num_processes} parallel processes...")
        
        # Start parallel processes
        for i in range(num_processes):
            result_queue = mp.Queue()
            control_value = mp.Value('i', 1)
            
            process = mp.Process(
                target=parallel_simulation_worker,
                args=(i, result_queue, control_value, 0.1 / self.speed_var.get())
            )
            process.start()
            
            self.parallel_processes.append(process)
            self.result_queues.append(result_queue)
            self.control_values.append(control_value)
            
            print(f"Started process {i} with PID {process.pid}")
        
        # Start result collection thread
        self.serial_thread = threading.Thread(target=self.collect_parallel_results)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        
        print("Parallel simulation setup complete")
    
    def run_serial_simulation(self):
        """Run serial simulation loop"""
        time_step = 0.1 / self.speed_var.get()
        
        while self.is_running:
            start_time = time.time()
            
            # Process all zones sequentially in this single thread
            self.serial_simulation.simulate_step(time_step)
            
            # Update current state for GUI
            self.current_state = self.serial_simulation.get_state()
            
            # Record performance
            execution_time = time.time() - start_time
            self.performance_data['serial']['execution_times'].append(execution_time)
            self.performance_data['serial']['total_steps'] += 1
            
            # Keep only last 100 measurements
            if len(self.performance_data['serial']['execution_times']) > 100:
                self.performance_data['serial']['execution_times'].pop(0)
            
            # Calculate average
            if self.performance_data['serial']['execution_times']:
                self.performance_data['serial']['avg_processing_time'] = \
                    sum(self.performance_data['serial']['execution_times']) / \
                    len(self.performance_data['serial']['execution_times'])
            
            # Control simulation speed
            sleep_time = max(0, time_step - execution_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def collect_parallel_results(self):
        """Collect and merge results from parallel simulation processes"""
        time_step = 0.1 / self.speed_var.get()
        collection_count = 0
        
        print("Starting parallel result collection...")
        
        while self.is_running:
            start_time = time.time()
            collection_count += 1
            
            # Get the best/latest result from any process
            best_state = None
            total_vehicles = 0
            active_processes = 0
            states_received = 0
            
            for i, result_queue in enumerate(self.result_queues):
                try:
                    # Check if process is still alive
                    if self.parallel_processes[i].is_alive():
                        active_processes += 1
                    
                    # Get latest state from this process
                    latest_state = None
                    while not result_queue.empty():
                        latest_state = result_queue.get_nowait()
                        states_received += 1
                    
                    if latest_state:
                        # Use the state with most vehicles as primary
                        vehicles_count = len(latest_state.get('vehicles', {}))
                        if vehicles_count > total_vehicles:
                            best_state = latest_state
                            total_vehicles = vehicles_count
                            
                except Exception as e:
                    if collection_count % 100 == 0:  # Print occasionally
                        print(f"Error collecting from process {i}: {e}")
                    continue  # Skip this process if error
            
            # Use the best state found
            if best_state:
                # Add parallel processing indicator
                best_state['processing_mode'] = 'parallel'
                best_state['active_processes'] = active_processes
                self.current_state = best_state
            
            # Debug output occasionally
            if collection_count % 100 == 0:
                print(f"Collection {collection_count}: Active processes: {active_processes}, States received: {states_received}, Best vehicles: {total_vehicles}")
            
            # Record performance
            execution_time = time.time() - start_time
            self.performance_data['parallel']['execution_times'].append(execution_time)
            self.performance_data['parallel']['total_steps'] += 1
            
            # Keep only last 100 measurements
            if len(self.performance_data['parallel']['execution_times']) > 100:
                self.performance_data['parallel']['execution_times'].pop(0)
            
            # Calculate average
            if self.performance_data['parallel']['execution_times']:
                self.performance_data['parallel']['avg_processing_time'] = \
                    sum(self.performance_data['parallel']['execution_times']) / \
                    len(self.performance_data['parallel']['execution_times'])
            
            time.sleep(max(0, time_step - execution_time))
        
        print("Parallel result collection stopped")
    
    def merge_zone_states(self, zone_states: List[Dict]) -> Dict:
        """Merge multiple zone states into a single state"""
        merged_state = {
            'vehicles': {},
            'traffic_lights': {},
            'roads': {},
            'zone_metrics': {'North': {}, 'South': {}, 'East': {}, 'West': {}},
            'total_vehicles': 0,
            'total_generated': 0,
            'total_completed': 0,
            'simulation_time': 0.0
        }
        
        # Merge data from all zones
        for zone_state in zone_states:
            if zone_state:
                # Merge vehicles (with unique IDs per zone)
                zone_id = zone_state.get('zone_id', 0)
                for vehicle_id, vehicle_data in zone_state.get('vehicles', {}).items():
                    unique_id = f"{zone_id}_{vehicle_id}"
                    merged_state['vehicles'][unique_id] = vehicle_data
                
                # Merge traffic lights
                merged_state['traffic_lights'].update(zone_state.get('traffic_lights', {}))
                
                # Merge roads
                merged_state['roads'].update(zone_state.get('roads', {}))
                
                # Update zone metrics
                zone_focus = zone_state.get('zone_focus', 'North')
                merged_state['zone_metrics'][zone_focus] = zone_state.get('zone_metrics', {}).get(zone_focus, {})
                
                # Sum totals
                merged_state['total_vehicles'] += zone_state.get('total_vehicles', 0)
                merged_state['total_generated'] += zone_state.get('total_generated', 0)
                merged_state['total_completed'] += zone_state.get('total_completed', 0)
                merged_state['simulation_time'] = max(merged_state['simulation_time'], 
                                                    zone_state.get('simulation_time', 0))
        
        return merged_state
    
    def stop_simulation(self):
        """Stop the simulation"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop serial simulation
        if self.serial_thread:
            # Thread will stop when is_running becomes False
            pass
        
        # Stop parallel processes
        if self.parallel_processes:
            print("Stopping parallel processes...")
            
            # Signal all processes to stop
            for i, control_value in enumerate(self.control_values):
                control_value.value = 0
                print(f"Signaled process {i} to stop")
            
            # Wait for processes to finish
            for i, process in enumerate(self.parallel_processes):
                print(f"Waiting for process {i} (PID {process.pid}) to finish...")
                process.join(timeout=2)
                if process.is_alive():
                    print(f"Process {i} still alive, terminating...")
                    process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        print(f"Process {i} still alive after terminate, killing...")
                        process.kill()
                else:
                    print(f"Process {i} finished gracefully")
        
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Stopped")
        
        self.log_message(f"{self.simulation_mode.title()} simulation stopped")
        
        # Show performance comparison if both modes have data
        self.show_performance_comparison()
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.stop_simulation()
        
        # Reset simulation objects
        self.serial_simulation = None
        self.current_state = None
        
        # Clear parallel process lists
        self.parallel_processes = []
        self.result_queues = []
        self.control_values = []
        
        # Reset performance data
        self.performance_data = {
            'serial': {'execution_times': [], 'avg_processing_time': 0, 'total_steps': 0},
            'parallel': {'execution_times': [], 'avg_processing_time': 0, 'total_steps': 0}
        }
        
        # Reset display
        self.draw_road_network()
        self.metrics_text.delete(1.0, tk.END)
        self.vehicle_text.delete(1.0, tk.END)
        self.perf_text.delete(1.0, tk.END)
        self.comparison_text.delete(1.0, tk.END)
        
        self.log_message("Simulation reset")
    
    def update_gui(self):
        """Update GUI periodically"""
        if self.is_running:
            # For serial mode, current_state is updated directly
            # For parallel mode, current_state is updated by collect_parallel_results
            if self.current_state:
                self.update_visualization()
                self.update_metrics()
                self.update_performance()
        
        # Always update comparison tab
        self.update_comparison_tab()
        
        # Schedule next update
        self.root.after(50, self.update_gui)  # 20 FPS
    
    def update_visualization(self):
        """Update the visual elements"""
        if not self.current_state:
            return
        
        # Clear previous vehicles and traffic lights
        self.canvas.delete("vehicle")
        self.canvas.delete("light_state")
        
        # Draw vehicles - appropriate size for larger network
        for vehicle_id, vehicle_data in self.current_state['vehicles'].items():
            pos = vehicle_data['position']
            color = vehicle_data['color']
            
            # Draw vehicle as a small rectangle
            size = 5  # Increased from 4 to 5
            self.canvas.create_rectangle(
                pos['x'] - size, pos['y'] - size,
                pos['x'] + size, pos['y'] + size,
                fill=color, outline="black", width=1, tags="vehicle"
            )
            
            # Optional: Add direction indicator
            self.canvas.create_oval(
                pos['x'] - 2, pos['y'] - 2,  # Increased from 1 to 2
                pos['x'] + 2, pos['y'] + 2,
                fill="white", tags="vehicle"
            )
        
        # Update traffic lights
        for light_id, light_data in self.current_state['traffic_lights'].items():
            pos = light_data['position']
            state = light_data['state']
            color = "green" if state == "GREEN" else "red"
            
            # Update traffic light color
            light_items = self.canvas.find_withtag(f"traffic_light_{light_id}")
            for item in light_items:
                self.canvas.itemconfig(item, fill=color)
        
        # Update zone colors based on congestion
        for zone_name, metrics in self.current_state['zone_metrics'].items():
            congestion = metrics['congestion']
            
            # Determine zone color
            if congestion < 0.3:
                zone_color = "lightgreen"
            elif congestion < 0.7:
                zone_color = "yellow"
            else:
                zone_color = "red"
            
            # Update zone circle color (make it more visible) - match the thicker borders
            zone_items = self.canvas.find_withtag(f"zone_{zone_name}")
            for item in zone_items:
                self.canvas.itemconfig(item, outline=zone_color, width=4)  # Increased from 3 to 4
    
    def update_metrics(self):
        """Update metrics display"""
        if not self.current_state:
            return
        
        self.metrics_text.delete(1.0, tk.END)
        
        # Processing mode indicator
        self.metrics_text.insert(tk.END, f"=== {self.simulation_mode.upper()} PROCESSING MODE ===\n")
        self.metrics_text.insert(tk.END, f"Mode: {self.simulation_mode.title()} Processing\n")
        
        if self.simulation_mode == "parallel":
            active_processes = len([p for p in self.parallel_processes if p.is_alive()])
            process_id = self.current_state.get('process_id', 'Unknown')
            self.metrics_text.insert(tk.END, f"Active Processes: {active_processes}\n")
            self.metrics_text.insert(tk.END, f"Displaying from Process: {process_id}\n")
        
        self.metrics_text.insert(tk.END, "\n")
        
        # Zone metrics
        self.metrics_text.insert(tk.END, "=== ZONE METRICS ===\n")
        zone_metrics = self.current_state.get('zone_metrics', {})
        for zone_name in ['North', 'South', 'East', 'West']:
            metrics = zone_metrics.get(zone_name, {'vehicles': 0, 'congestion': 0.0, 'avg_speed': 0.0})
            self.metrics_text.insert(tk.END, f"\n{zone_name} Zone:\n")
            self.metrics_text.insert(tk.END, f"  Vehicles: {metrics.get('vehicles', 0)}\n")
            self.metrics_text.insert(tk.END, f"  Congestion: {metrics.get('congestion', 0):.1%}\n")
            self.metrics_text.insert(tk.END, f"  Avg Speed: {metrics.get('avg_speed', 0):.1f} km/h\n")
        
        # Overall stats
        self.metrics_text.insert(tk.END, "\n=== OVERALL STATISTICS ===\n")
        self.metrics_text.insert(tk.END, f"Active Vehicles: {self.current_state.get('total_vehicles', 0)}\n")
        self.metrics_text.insert(tk.END, f"Total Generated: {self.current_state.get('total_generated', 0)}\n")
        self.metrics_text.insert(tk.END, f"Total Completed: {self.current_state.get('total_completed', 0)}\n")
        self.metrics_text.insert(tk.END, f"Simulation Time: {self.current_state.get('simulation_time', 0):.1f}s\n")
        
        # Vehicle details
        self.vehicle_text.delete(1.0, tk.END)
        self.vehicle_text.insert(tk.END, f"=== ACTIVE VEHICLES ({self.simulation_mode.upper()}) ===\n\n")
        
        vehicle_count = 0
        for vehicle_id, vehicle_data in self.current_state.get('vehicles', {}).items():
            if vehicle_count >= 8:  # Show first 8
                break
            self.vehicle_text.insert(tk.END, f"Vehicle {vehicle_id}:\n")
            self.vehicle_text.insert(tk.END, f"  Route: {vehicle_data.get('origin', 'Unknown')} → {vehicle_data.get('destination', 'Unknown')}\n")
            self.vehicle_text.insert(tk.END, f"  Speed: {vehicle_data.get('speed', 0):.1f} km/h\n\n")
            vehicle_count += 1
    
    def update_performance(self):
        """Update performance metrics"""
        current_time = time.time()
        self.frame_count += 1
        
        # Calculate FPS every second
        if current_time - self.last_update_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_update_time)
            self.frame_count = 0
            self.last_update_time = current_time
            
            # Update performance display
            self.perf_text.delete(1.0, tk.END)
            self.perf_text.insert(tk.END, "=== PERFORMANCE METRICS ===\n\n")
            
            # Current mode performance
            current_mode = self.simulation_mode
            if self.performance_data[current_mode]['execution_times']:
                avg_time = self.performance_data[current_mode]['avg_processing_time']
                steps = self.performance_data[current_mode]['total_steps']
                
                self.perf_text.insert(tk.END, f"Current Mode: {current_mode.title()}\n")
                self.perf_text.insert(tk.END, f"  Avg Step Time: {avg_time:.4f}s\n")
                self.perf_text.insert(tk.END, f"  Steps Completed: {steps}\n")
                self.perf_text.insert(tk.END, f"  GUI FPS: {fps:.1f}\n\n")
            
            # Performance comparison
            serial_avg = self.performance_data['serial']['avg_processing_time']
            parallel_avg = self.performance_data['parallel']['avg_processing_time']
            
            if serial_avg > 0 and parallel_avg > 0:
                speedup = serial_avg / parallel_avg
                self.perf_text.insert(tk.END, "=== PERFORMANCE COMPARISON ===\n")
                self.perf_text.insert(tk.END, f"Serial Avg: {serial_avg:.4f}s\n")
                self.perf_text.insert(tk.END, f"Parallel Avg: {parallel_avg:.4f}s\n")
                self.perf_text.insert(tk.END, f"Speedup: {speedup:.2f}x\n")
                
                if speedup > 1:
                    self.perf_text.insert(tk.END, f"✓ Parallel is {speedup:.2f}x faster!\n")
                else:
                    self.perf_text.insert(tk.END, f"⚠ Serial is {1/speedup:.2f}x faster\n")
            
            if self.current_state:
                self.perf_text.insert(tk.END, f"\nSimulation Speed: {self.speed_var.get():.1f}x\n")
                self.perf_text.insert(tk.END, f"Active Vehicles: {len(self.current_state.get('vehicles', {}))}\n")
    
    def show_performance_comparison(self):
        """Show detailed performance comparison"""
        serial_data = self.performance_data['serial']
        parallel_data = self.performance_data['parallel']
        
        if serial_data['total_steps'] > 0 and parallel_data['total_steps'] > 0:
            comparison = f"""
=== FINAL PERFORMANCE COMPARISON ===

Serial Processing:
  - Total Steps: {serial_data['total_steps']}
  - Average Time: {serial_data['avg_processing_time']:.4f}s per step
  - Total Execution Time: {sum(serial_data['execution_times']):.2f}s

Parallel Processing:
  - Total Steps: {parallel_data['total_steps']}
  - Average Time: {parallel_data['avg_processing_time']:.4f}s per step
  - Total Execution Time: {sum(parallel_data['execution_times']):.2f}s

Performance Gain:
  - Speedup: {serial_data['avg_processing_time'] / parallel_data['avg_processing_time']:.2f}x
  - Efficiency: {(serial_data['avg_processing_time'] / parallel_data['avg_processing_time'] / 4 * 100):.1f}% (4 cores)
            """
            
            self.log_message("Performance comparison available - check console output")
            print(comparison)
    
    def update_comparison_tab(self):
        """Update the comparison tab with current performance data"""
        self.comparison_text.delete(1.0, tk.END)
        
        self.comparison_text.insert(tk.END, "=== REAL-TIME PERFORMANCE COMPARISON ===\n\n")
        
        # Current status
        self.comparison_text.insert(tk.END, f"Current Mode: {self.simulation_mode.title()}\n")
        self.comparison_text.insert(tk.END, f"Simulation Running: {'Yes' if self.is_running else 'No'}\n\n")
        
        # Serial performance
        serial_data = self.performance_data['serial']
        self.comparison_text.insert(tk.END, "📊 SERIAL PROCESSING:\n")
        if serial_data['total_steps'] > 0:
            self.comparison_text.insert(tk.END, f"  ✓ Steps Completed: {serial_data['total_steps']}\n")
            self.comparison_text.insert(tk.END, f"  ✓ Avg Step Time: {serial_data['avg_processing_time']:.4f}s\n")
            self.comparison_text.insert(tk.END, f"  ✓ Total Runtime: {sum(serial_data['execution_times']):.2f}s\n")
            if len(serial_data['execution_times']) > 0:
                min_time = min(serial_data['execution_times'])
                max_time = max(serial_data['execution_times'])
                self.comparison_text.insert(tk.END, f"  ✓ Min/Max: {min_time:.4f}s / {max_time:.4f}s\n")
        else:
            self.comparison_text.insert(tk.END, "  ⚠ No data yet - run Serial mode first\n")
        
        self.comparison_text.insert(tk.END, "\n")
        
        # Parallel performance
        parallel_data = self.performance_data['parallel']
        self.comparison_text.insert(tk.END, "⚡ PARALLEL PROCESSING:\n")
        if parallel_data['total_steps'] > 0:
            self.comparison_text.insert(tk.END, f"  ✓ Steps Completed: {parallel_data['total_steps']}\n")
            self.comparison_text.insert(tk.END, f"  ✓ Avg Step Time: {parallel_data['avg_processing_time']:.4f}s\n")
            self.comparison_text.insert(tk.END, f"  ✓ Total Runtime: {sum(parallel_data['execution_times']):.2f}s\n")
            if len(parallel_data['execution_times']) > 0:
                min_time = min(parallel_data['execution_times'])
                max_time = max(parallel_data['execution_times'])
                self.comparison_text.insert(tk.END, f"  ✓ Min/Max: {min_time:.4f}s / {max_time:.4f}s\n")
            if self.is_running and self.simulation_mode == "parallel":
                active_processes = len([p for p in self.parallel_processes if p.is_alive()])
                self.comparison_text.insert(tk.END, f"  ✓ Active Processes: {active_processes}/4\n")
        else:
            self.comparison_text.insert(tk.END, "  ⚠ No data yet - run Parallel mode first\n")
        
        self.comparison_text.insert(tk.END, "\n")
        
        # Performance comparison
        if (serial_data['avg_processing_time'] > 0 and 
            parallel_data['avg_processing_time'] > 0):
            
            speedup = serial_data['avg_processing_time'] / parallel_data['avg_processing_time']
            efficiency = (speedup / 4) * 100  # 4 cores
            
            self.comparison_text.insert(tk.END, "🏆 PERFORMANCE ANALYSIS:\n")
            self.comparison_text.insert(tk.END, f"  🚀 Speedup: {speedup:.2f}x\n")
            self.comparison_text.insert(tk.END, f"  📈 Efficiency: {efficiency:.1f}% (4 cores)\n")
            
            if speedup > 1.5:
                self.comparison_text.insert(tk.END, f"  ✅ Parallel is significantly faster!\n")
            elif speedup > 1.0:
                self.comparison_text.insert(tk.END, f"  ✅ Parallel is faster\n")
            else:
                self.comparison_text.insert(tk.END, f"  ⚠ Serial is faster (overhead issues?)\n")
            
            # Recommendations
            self.comparison_text.insert(tk.END, "\n💡 RECOMMENDATIONS:\n")
            if speedup > 2.0:
                self.comparison_text.insert(tk.END, "  • Excellent parallelization! Consider more complex tasks.\n")
            elif speedup > 1.2:
                self.comparison_text.insert(tk.END, "  • Good parallelization. Normal for this workload.\n")
            else:
                self.comparison_text.insert(tk.END, "  • Limited speedup. Task may be too simple for parallelization.\n")
        else:
            self.comparison_text.insert(tk.END, "🔄 Run both Serial and Parallel modes to see comparison\n")
    
    def show_comparison_window(self):
        """Show detailed performance comparison in a separate window"""
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Performance Comparison Analysis")
        comparison_window.geometry("900x600")  # Reduced from 1000x700
        
        # Create notebook for different comparison views
        notebook = ttk.Notebook(comparison_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Detailed statistics tab
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="Detailed Statistics")
        
        stats_text = scrolledtext.ScrolledText(stats_frame, font=("Courier", 10))
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Charts tab
        charts_frame = ttk.Frame(notebook)
        notebook.add(charts_frame, text="Performance Charts")
        
        # Create matplotlib figure for charts
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 6))  # Reduced from (12, 8)
            canvas = FigureCanvasTkAgg(fig, charts_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Chart 1: Execution Time Comparison
            serial_times = self.performance_data['serial']['execution_times']
            parallel_times = self.performance_data['parallel']['execution_times']
            
            if serial_times and parallel_times:
                ax1.plot(serial_times[-50:], label='Serial', color='blue', linewidth=2)
                ax1.plot(parallel_times[-50:], label='Parallel', color='red', linewidth=2)
                ax1.set_title('Execution Times (Last 50 Steps)')
                ax1.set_xlabel('Step Number')
                ax1.set_ylabel('Time (seconds)')
                ax1.legend()
                ax1.grid(True)
            
            # Chart 2: Average Performance Bar Chart
            modes = []
            avg_times = []
            colors = []
            
            if self.performance_data['serial']['avg_processing_time'] > 0:
                modes.append('Serial')
                avg_times.append(self.performance_data['serial']['avg_processing_time'])
                colors.append('blue')
            
            if self.performance_data['parallel']['avg_processing_time'] > 0:
                modes.append('Parallel')
                avg_times.append(self.performance_data['parallel']['avg_processing_time'])
                colors.append('red')
            
            if modes:
                bars = ax2.bar(modes, avg_times, color=colors, alpha=0.7)
                ax2.set_title('Average Execution Time Comparison')
                ax2.set_ylabel('Time (seconds)')
                
                # Add value labels on bars
                for bar, time in zip(bars, avg_times):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{time:.4f}s', ha='center', va='bottom')
            
            # Chart 3: Speedup Chart
            if (self.performance_data['serial']['avg_processing_time'] > 0 and 
                self.performance_data['parallel']['avg_processing_time'] > 0):
                
                speedup = (self.performance_data['serial']['avg_processing_time'] / 
                          self.performance_data['parallel']['avg_processing_time'])
                
                ax3.bar(['Actual Speedup'], [speedup], color='green', alpha=0.7)
                ax3.axhline(y=1, color='black', linestyle='--', label='No Speedup')
                ax3.axhline(y=4, color='orange', linestyle='--', label='Ideal (4 cores)')
                ax3.set_title('Parallel Processing Speedup')
                ax3.set_ylabel('Speedup Factor')
                ax3.legend()
                ax3.text(0, speedup + 0.1, f'{speedup:.2f}x', ha='center', va='bottom', 
                        fontweight='bold', fontsize=12)
            
            # Chart 4: Efficiency Chart
            if (self.performance_data['serial']['avg_processing_time'] > 0 and 
                self.performance_data['parallel']['avg_processing_time'] > 0):
                
                speedup = (self.performance_data['serial']['avg_processing_time'] / 
                          self.performance_data['parallel']['avg_processing_time'])
                efficiency = (speedup / 4) * 100
                
                ax4.pie([efficiency, 100-efficiency], labels=['Efficiency', 'Overhead'], 
                       colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
                ax4.set_title(f'Parallel Efficiency\n({efficiency:.1f}% of ideal)')
            
            plt.tight_layout()
            canvas.draw()
            
        except Exception as e:
            error_label = tk.Label(charts_frame, text=f"Charts unavailable: {str(e)}", 
                                 font=("Arial", 12), fg="red")
            error_label.pack(expand=True)
        
        # Fill detailed statistics
        self.fill_detailed_statistics(stats_text)
        
        # Refresh button
        refresh_frame = ttk.Frame(comparison_window)
        refresh_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        ttk.Button(refresh_frame, text="Refresh Data", 
                  command=lambda: self.refresh_comparison_data(stats_text, canvas if 'canvas' in locals() else None)).pack(side=tk.LEFT)
        
        ttk.Button(refresh_frame, text="Export Results", 
                  command=self.export_performance_results).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(refresh_frame, text="Close", 
                  command=comparison_window.destroy).pack(side=tk.RIGHT)
    
    def fill_detailed_statistics(self, text_widget):
        """Fill the detailed statistics text widget"""
        text_widget.delete(1.0, tk.END)
        
        text_widget.insert(tk.END, "╔" + "="*70 + "╗\n")
        text_widget.insert(tk.END, "║" + " DETAILED PERFORMANCE COMPARISON REPORT ".center(70) + "║\n")
        text_widget.insert(tk.END, "╚" + "="*70 + "╝\n\n")
        
        # System information
        text_widget.insert(tk.END, "SYSTEM INFORMATION:\n")
        text_widget.insert(tk.END, "-" * 20 + "\n")
        text_widget.insert(tk.END, f"CPU Cores Used: 4 (parallel mode)\n")
        text_widget.insert(tk.END, f"Simulation Speed: {self.speed_var.get():.1f}x\n")
        text_widget.insert(tk.END, f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Serial mode statistics
        serial_data = self.performance_data['serial']
        text_widget.insert(tk.END, "SERIAL PROCESSING STATISTICS:\n")
        text_widget.insert(tk.END, "-" * 35 + "\n")
        
        if serial_data['total_steps'] > 0:
            serial_times = serial_data['execution_times']
            text_widget.insert(tk.END, f"Total Steps:          {serial_data['total_steps']:,}\n")
            text_widget.insert(tk.END, f"Average Step Time:    {serial_data['avg_processing_time']:.6f} seconds\n")
            text_widget.insert(tk.END, f"Minimum Step Time:    {min(serial_times):.6f} seconds\n")
            text_widget.insert(tk.END, f"Maximum Step Time:    {max(serial_times):.6f} seconds\n")
            text_widget.insert(tk.END, f"Standard Deviation:   {np.std(serial_times):.6f} seconds\n")
            text_widget.insert(tk.END, f"Total Runtime:        {sum(serial_times):.3f} seconds\n")
            text_widget.insert(tk.END, f"Steps per Second:     {len(serial_times) / sum(serial_times):.2f}\n")
        else:
            text_widget.insert(tk.END, "No serial mode data available.\n")
        
        text_widget.insert(tk.END, "\n")
        
        # Parallel mode statistics
        parallel_data = self.performance_data['parallel']
        text_widget.insert(tk.END, "PARALLEL PROCESSING STATISTICS:\n")
        text_widget.insert(tk.END, "-" * 37 + "\n")
        
        if parallel_data['total_steps'] > 0:
            parallel_times = parallel_data['execution_times']
            text_widget.insert(tk.END, f"Total Steps:          {parallel_data['total_steps']:,}\n")
            text_widget.insert(tk.END, f"Average Step Time:    {parallel_data['avg_processing_time']:.6f} seconds\n")
            text_widget.insert(tk.END, f"Minimum Step Time:    {min(parallel_times):.6f} seconds\n")
            text_widget.insert(tk.END, f"Maximum Step Time:    {max(parallel_times):.6f} seconds\n")
            text_widget.insert(tk.END, f"Standard Deviation:   {np.std(parallel_times):.6f} seconds\n")
            text_widget.insert(tk.END, f"Total Runtime:        {sum(parallel_times):.3f} seconds\n")
            text_widget.insert(tk.END, f"Steps per Second:     {len(parallel_times) / sum(parallel_times):.2f}\n")
            text_widget.insert(tk.END, f"Processes Used:       4\n")
        else:
            text_widget.insert(tk.END, "No parallel mode data available.\n")
        
        text_widget.insert(tk.END, "\n")
        
        # Comparison analysis
        if (serial_data['avg_processing_time'] > 0 and 
            parallel_data['avg_processing_time'] > 0):
            
            speedup = serial_data['avg_processing_time'] / parallel_data['avg_processing_time']
            efficiency = (speedup / 4) * 100
            
            text_widget.insert(tk.END, "PERFORMANCE ANALYSIS:\n")
            text_widget.insert(tk.END, "-" * 22 + "\n")
            text_widget.insert(tk.END, f"Speedup Factor:       {speedup:.3f}x\n")
            text_widget.insert(tk.END, f"Parallel Efficiency:  {efficiency:.2f}%\n")
            text_widget.insert(tk.END, f"Time Saved:           {(serial_data['avg_processing_time'] - parallel_data['avg_processing_time']) * 1000:.2f} ms per step\n")
            
            if speedup > 3.0:
                performance_rating = "Excellent"
                recommendation = "Outstanding parallelization! Consider even more complex tasks."
            elif speedup > 2.0:
                performance_rating = "Very Good"
                recommendation = "Great parallelization. Well-suited for this workload."
            elif speedup > 1.5:
                performance_rating = "Good"
                recommendation = "Solid performance improvement. Normal for this type of simulation."
            elif speedup > 1.0:
                performance_rating = "Fair"
                recommendation = "Some improvement. Task may have limited parallelization potential."
            else:
                performance_rating = "Poor"
                recommendation = "Serial processing is faster. Task not suitable for parallelization."
            
            text_widget.insert(tk.END, f"Performance Rating:   {performance_rating}\n")
            text_widget.insert(tk.END, f"\nRECOMMENDATION:\n{recommendation}\n")
            
            # Theoretical analysis
            text_widget.insert(tk.END, "\nTHEORETICAL ANALYSIS:\n")
            text_widget.insert(tk.END, "-" * 22 + "\n")
            text_widget.insert(tk.END, f"Amdahl's Law Efficiency: {efficiency:.2f}% of ideal\n")
            text_widget.insert(tk.END, f"Parallel Overhead:       {(1 - efficiency/100) * 100:.2f}%\n")
            
            # Calculate theoretical maximum speedup
            parallel_fraction = 0.8  # Assume 80% of code is parallelizable
            amdahl_speedup = 1 / ((1 - parallel_fraction) + (parallel_fraction / 4))
            text_widget.insert(tk.END, f"Theoretical Max (80% ||): {amdahl_speedup:.2f}x\n")
        
        text_widget.insert(tk.END, "\n" + "="*72 + "\n")
        text_widget.insert(tk.END, "Report generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    
    def refresh_comparison_data(self, stats_text, canvas):
        """Refresh the comparison data"""
        self.fill_detailed_statistics(stats_text)
        if canvas:
            try:
                canvas.draw()
            except:
                pass
    
    def export_performance_results(self):
        """Export performance results to a file"""
        try:
            filename = f"traffic_simulation_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(filename, 'w') as f:
                f.write("TRAFFIC SIMULATION PERFORMANCE REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Write all performance data
                f.write("SERIAL MODE DATA:\n")
                serial_data = self.performance_data['serial']
                f.write(f"Steps: {serial_data['total_steps']}\n")
                f.write(f"Avg Time: {serial_data['avg_processing_time']:.6f}s\n")
                f.write(f"Execution Times: {serial_data['execution_times']}\n\n")
                
                f.write("PARALLEL MODE DATA:\n")
                parallel_data = self.performance_data['parallel']
                f.write(f"Steps: {parallel_data['total_steps']}\n")
                f.write(f"Avg Time: {parallel_data['avg_processing_time']:.6f}s\n")
                f.write(f"Execution Times: {parallel_data['execution_times']}\n\n")
                
                if (serial_data['avg_processing_time'] > 0 and 
                    parallel_data['avg_processing_time'] > 0):
                    speedup = serial_data['avg_processing_time'] / parallel_data['avg_processing_time']
                    f.write(f"SPEEDUP: {speedup:.3f}x\n")
                    f.write(f"EFFICIENCY: {(speedup/4)*100:.2f}%\n")
            
            messagebox.showinfo("Export Successful", f"Performance data exported to {filename}")
            self.log_message(f"Performance data exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export data: {str(e)}")
            self.log_message(f"Export failed: {str(e)}")
    
    def log_message(self, message: str):
        """Log a message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

def main():
    """Main application entry point"""
    mp.set_start_method('spawn', force=True)
    
    root = tk.Tk()
    app = TrafficSimulationGUI(root)
    
    def on_closing():
        app.stop_simulation()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()