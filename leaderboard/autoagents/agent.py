import numpy as np
import carla
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


def get_entry_point():
    return 'MyAgent'


class MyAgent(AutonomousAgent):
    def setup(self, path_to_conf_file=None):
        """
        Inizializzazione dell'agente. Imposta il tipo di track.
        """
        self.track = Track.SENSORS  # o Track.MAP se vuoi usare HD map
        print("MyAgent setup completed.")

    def sensors(self):
        """
        Definisce i sensori richiesti dall'agente.
        """
        sensors = [
            {
                'type': 'sensor.camera.rgb',
                'id': 'Center',
                'x': 1.2, 'y': 0.0, 'z': 1.8,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 800, 'height': 600, 'fov': 90
            },
            {
                'type': 'sensor.other.gnss',
                'id': 'GPS',
                'x': 0.7, 'y': 0.0, 'z': 1.6
            },
            {
                'type': 'sensor.speedometer',
                'id': 'Speed'
            }
        ]
        return sensors

    def run_step(self, input_data, timestamp):
        """
        Logica principale di controllo per ogni frame.
        Restituisce carla.VehicleControl().
        """
        control = carla.VehicleControl()

        # Recupera dati GPS e velocità
        gps = input_data['GPS'][1]  # (lat, lon, alt)
        speed = input_data['Speed'][1]['speed']

        # Recupera il prossimo waypoint dalla rotta globale
        waypoint = self._get_next_waypoint(gps)
        steer = self._compute_steer(waypoint, gps)

        # Controllo velocità semplice
        target_speed = 20.0  # km/h
        throttle, brake = self._compute_throttle_brake(speed, target_speed)

        control.steer = steer
        control.throttle = throttle
        control.brake = brake

        return control

    def _get_next_waypoint(self, gps):
        """
        Restituisce il prossimo waypoint dalla rotta globale.
        Ogni elemento di _global_plan ha tipicamente forma:
        ((lat, lon), RoadOption)
        """
        if not hasattr(self, '_global_plan') or len(self._global_plan) == 0:
            return None
        return self._global_plan[0]


    def _compute_steer(self, waypoint, gps):
        """
        Calcola uno sterzo molto semplice usando GPS corrente e prossimo punto della global plan.
        """
        if waypoint is None:
            return 0.0

        target_gps = waypoint[0]   # prende (lat, lon)

        curr_lat, curr_lon, _ = gps
        tgt_lat, tgt_lon = target_gps

        dlat = tgt_lat - curr_lat
        dlon = tgt_lon - curr_lon

        steer = (dlon * 2.0) - (dlat * 0.5)
        return float(np.clip(steer, -1.0, 1.0))

    def _compute_throttle_brake(self, speed, target_speed):
        """
        Controllo semplice della velocità.
        """
        if speed < target_speed:
            return 0.5, 0.0  # throttle, brake
        else:
            return 0.0, 0.2  # riduci velocità

    def destroy(self):
        pass