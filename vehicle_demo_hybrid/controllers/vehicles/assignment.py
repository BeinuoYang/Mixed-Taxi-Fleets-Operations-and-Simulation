import numpy as np
import requests
import gurobipy as gp
from gurobipy import GRB

from .tasks import VehicleTasks
from ..pedestrians.agents import PedestrianStates
from ..vehicles.states import VehicleStates
from ..network.networks import Network

def m_to_km(m):
    """Converts metres to kilometres"""
    return m / 1000

def hours_to_seconds(hours):
    """Converts hours to seconds"""
    return hours * 60 * 60

class Assignment:
    @staticmethod
    def step_assignment(pedestrians, vehicles, parkinglots,
                        network: Network, current_time, maxRuntime=60):
        speed = vehicles.spd_kmh[0]

        peds_to_assign = pedestrians.index_by_state(PedestrianStates.Unassigned)
        orders_num = len(peds_to_assign)
        if orders_num == 0:
            return
        pickup_node_ids = np.arange(0, orders_num)
        pl_num = len(parkinglots.id)
        parkinglot_ids = np.arange(0, pl_num)

        # online vehicles = idle vehicles + charging AETs + moving to parkinglot AETs + queuing AETs
        idle_vehicles_origin = vehicles.index_by_state(VehicleStates.Idle)
        charging_vehicles = vehicles.index_by_state(VehicleStates.Charging)
        charging_AET = [i for i in charging_vehicles if vehicles.is_autonomous[i]]
        moving_vehicles = vehicles.index_by_state(VehicleStates.Moving)
        moving_to_pl_AET = [i for i in moving_vehicles if (vehicles.is_autonomous[i] and vehicles.is_electric[i]
                            and (type(vehicles.task_queue[i][0]) is VehicleTasks.Charging))]
        queuing_vehicles = vehicles.index_by_state(VehicleStates.Queuing)
        queuing_AET = [i for i in queuing_vehicles if vehicles.is_autonomous[i]]
        vehicle_num = len(idle_vehicles_origin)

        # There are four types of vehicles:
        # 1.AET: Autonomous, Electric 2.HET: Human-driven, Electric 4.HGT: Human-driven, Gasoline
        idle_AET, HET, HGT = [], [], []
        idle_vehicles = []
        i, j = 0, 0
        while i < vehicle_num:
            veh_idx = idle_vehicles_origin[j]
            veh_id = vehicles.id[veh_idx]
            if vehicles.is_electric[veh_idx]:
                if vehicles.is_autonomous[veh_idx]:
                    idle_vehicles.append(veh_idx)
                    idle_AET.append(i)
                    i += 1
                else:
                    min_battery_threshold = 0.75 - vehicles.range_anxiety[veh_idx] * 0.05
                    if vehicles.battery_level[veh_idx] < vehicles.battery_capacity[veh_idx]*min_battery_threshold:
                        # offline to charge
                        vehicle_num -= 1
                        vehicles.is_online[veh_idx] = False
                        dist_to_pl = np.zeros(pl_num)
                        for pl_idx in range(pl_num):
                            try:
                                dist_to_pl[pl_idx] = network.shortest_path_cost(vehicles.current_node[veh_idx], parkinglots.node_id[pl_idx])
                            except KeyError:
                                dist_to_pl[pl_idx] = 100000
                        dist_to_pl = m_to_km(dist_to_pl)
                        pl_idx = np.argmin(dist_to_pl)
                        task1 = vehicles.task_container.generate_task(
                            VehicleTasks.OfflineToParkingLot,
                            veh_id,
                            "destination_long", parkinglots.lon_deg[pl_idx],
                            "destination_lat", parkinglots.lat_deg[pl_idx],
                            "destination_node", parkinglots.node_id[pl_idx],
                        )
                        max_battery_threshold = np.random.normal(0.85, 0.3)
                        while ((max_battery_threshold < min_battery_threshold) or (max_battery_threshold > 1) or
                               (max_battery_threshold < 0.7)):
                            max_battery_threshold = np.random.normal(0.85, 0.3)
                        charge_duration = hours_to_seconds(((vehicles.battery_capacity[veh_idx] * max_battery_threshold
                                                             - vehicles.battery_level[veh_idx] +
                                                             dist_to_pl[pl_idx] * vehicles.energy_consumption[veh_idx])
                                                            / vehicles.charge_rate[veh_idx]))
                        task2 = vehicles.task_container.generate_task(
                            VehicleTasks.OfflineCharging,
                            veh_id,
                            "duration", charge_duration
                        )
                        vehicles.add_tasks_by_id(veh_id, task1)
                        vehicles.add_tasks_by_id(veh_id, task2)
                        j += 1
                        continue
                    elif vehicles.cont_op_time[veh_idx] > hours_to_seconds(vehicles.max_cont_op_time[veh_idx]):
                        # finished work
                        vehicle_num -= 1
                        vehicles.is_online[veh_idx] = False
                        vehicles.state_container.change_agent_state([veh_id], VehicleStates.Finished)
                        j += 1
                        continue
                    else:
                        idle_vehicles.append(veh_idx)
                        HET.append(i)
                        i += 1
            else:
                if vehicles.cont_op_time[veh_idx] > hours_to_seconds(vehicles.max_cont_op_time[veh_idx]):
                    # finished work
                    vehicle_num -= 1
                    vehicles.is_online[veh_idx] = False
                    vehicles.state_container.change_agent_state([veh_id], VehicleStates.Finished)
                    j += 1
                    continue
                else:
                    idle_vehicles.append(veh_idx)
                    HGT.append(i)
                    i += 1
            j += 1

        # idle_AET_num = 0 if len(idle_AET) == 0 else idle_AET[-1]
        AET = idle_AET + [i+vehicle_num for i in range(len(charging_AET)+len(moving_to_pl_AET)+len(queuing_AET))]
        vehicle_num += len(charging_AET) + len(moving_to_pl_AET) + len(queuing_AET)
        if vehicle_num == 0:
            return
        idle_vehicles = np.array(idle_vehicles + charging_AET + moving_to_pl_AET + queuing_AET)
        vehicle_ids = np.arange(0, vehicle_num)

        AET_HET = AET + HET
        if len(AET_HET) > 0:
            u1 = vehicles.energy_consumption[idle_vehicles[AET_HET[0]]]
        else:
            u1 = 0

        # distance matrix
        # TODO: Use param to get the osrm_url?
        osrm_url = "http://localhost:{}/table/v1/driving/".format(network.owner.params['osrm_port'])
        # g1: vehicle2pickup g2: pickup2dropoff(no need to use matrix?) g3: dropoff2parkinglot
        coordinates1, coordinates2, coordinates3 = "", "", ""

        # TODO: Use get_location to get the coord of nodes?
        sources1, destinations1 = "", ""
        for v in vehicle_ids:
            sources1 = sources1 + ";" + str(v)
            veh_idx = idle_vehicles[v]
            if veh_idx in moving_to_pl_AET:
                coordinates1 = (coordinates1 + ";" +
                                str(network.graph.nodes[vehicles.pre_node[veh_idx]]['x']) + "," +
                                str(network.graph.nodes[vehicles.pre_node[veh_idx]]['y']))
            else:
                coordinates1 = (coordinates1 + ";" +
                                str(network.graph.nodes[vehicles.current_node[veh_idx]]['x']) + "," +
                                str(network.graph.nodes[vehicles.current_node[veh_idx]]['y']))
        for o in pickup_node_ids:
            destinations1 = destinations1 + ";" + str(vehicle_num + o)
            coordinates1 = (coordinates1 + ";" +
                            str(network.graph.nodes[pedestrians.pickup_node_id[peds_to_assign[o]]]['x']) + ","
                            + str(network.graph.nodes[pedestrians.pickup_node_id[peds_to_assign[o]]]['y']))
        sources1 = sources1[1:]
        destinations1 = destinations1[1:]
        coordinates1 = coordinates1[1:]
        request_url = f"{osrm_url}{coordinates1}?sources={sources1}&destinations={destinations1}&annotations=distance"
        response = requests.get(request_url)
        if response.status_code == 200:
            data = response.json()
            g1 = np.array(data['distances'])
            g1[g1 == None] = 999999
            g1 = m_to_km(g1)
        else:
            raise Exception(
                f"Error fetching data from OSRM for O-D pair: {response.status_code}")

        sources2, destinations2 = "", ""
        for o in pickup_node_ids:
            sources2 = sources2 + ";" + str(o)
            coordinates2 = (coordinates2 + ";" +
                            str(network.graph.nodes[pedestrians.pickup_node_id[peds_to_assign[o]]]['x']) + "," +
                            str(network.graph.nodes[pedestrians.pickup_node_id[peds_to_assign[o]]]['y']))
        for o in pickup_node_ids:
            destinations2 = destinations2 + ";" + str(orders_num + o)
            coordinates2 = (coordinates2 + ";" +
                            str(network.graph.nodes[pedestrians.dropoff_node_id[peds_to_assign[o]]]['x']) + "," +
                            str(network.graph.nodes[pedestrians.dropoff_node_id[peds_to_assign[o]]]['y']))
        sources2 = sources2[1:]
        destinations2 = destinations2[1:]
        coordinates2 = coordinates2[1:]
        request_url = f"{osrm_url}{coordinates2}?sources={sources2}&destinations={destinations2}&annotations=distance"
        response = requests.get(request_url)
        if response.status_code == 200:
            data = response.json()
            g2 = np.diag(np.array(data['distances']))
            g2 = m_to_km(g2)
        else:
            raise Exception(
                f"Error fetching data from OSRM for O-D pair: {response.status_code}")

        sources3, destinations3 = "", ""
        for d in pickup_node_ids:
            sources3 = sources3 + ';' + str(d)
            coordinates3 = (coordinates3 + ";" +
                            str(network.graph.nodes[pedestrians.dropoff_node_id[peds_to_assign[d]]]['x']) + "," +
                            str(network.graph.nodes[pedestrians.dropoff_node_id[peds_to_assign[d]]]['y']))
        for s in parkinglot_ids:
            destinations3 = destinations3 + ";" + str(orders_num + s)
            coordinates3 = (coordinates3 + ";" +
                            str(network.graph.nodes[parkinglots.node_id[s]]['x']) + "," +
                            str(network.graph.nodes[parkinglots.node_id[s]]['y']))
        sources3 = sources3[1:]
        destinations3 = destinations3[1:]
        coordinates3 = coordinates3[1:]
        request_url = f"{osrm_url}{coordinates3}?sources={sources3}&destinations={destinations3}&annotations=distance"
        response = requests.get(request_url)
        if response.status_code == 200:
            data = response.json()
            g3 = np.array(data['distances'])
            g3[g3 == None] = 999999
            g3 = m_to_km(g3)
        else:
            raise Exception(
                f"Error fetching data from OSRM for O-D pair: {response.status_code}")

        # time matrix
        h1 = hours_to_seconds(g1/speed)

        c1 = 14
        c2 = 7
        c3 = 4.031
        c4 = 6.176
        c5 = 7.121

        model = gp.Model()
        model.setParam('OutputFlag', 0)

        # decision variables
        x = model.addVars(vehicle_ids, pickup_node_ids, vtype=GRB.BINARY, name='vehicle_customer_assignment')

        model.update()

        obj = gp.quicksum((gp.quicksum((c1 + c2*g2[r] - c3*(g1[v, r]+g2[r])) * x[v, r] for v in AET) +
                           gp.quicksum((c1 + c2*g2[r] - c4*(g1[v, r]+g2[r])) * x[v, r] for v in HET) +
                           gp.quicksum((c1 + c2*g2[r] - c5*(g1[v, r]+g2[r])) * x[v, r] for v in HGT))
                          for r in pickup_node_ids)
        model.setObjective(obj, GRB.MAXIMIZE)

        # Constraint (2)
        model.addConstrs((gp.quicksum(x[v, r] for v in vehicle_ids) <= 1 for r in pickup_node_ids),
                         name='constrain(2-1)')

        model.addConstrs((gp.quicksum(x[v, r] for r in pickup_node_ids) <= 1 for v in vehicle_ids),
                         name='constrain(2-2)')

        # Constraint (3)
        model.addConstrs(
            (x[v, r] * (current_time + h1[v, r]) - pedestrians.pickup_tw_end[peds_to_assign[r]] <= 0
             for v in vehicle_ids
             for r in pickup_node_ids),
            name='constrain(3)'
        )

        # Constraint (4)
        min_s_dist = {r: min(g3[r, s] for s in parkinglot_ids) for r in pickup_node_ids}
        model.addConstrs(
            (gp.quicksum(
                u1 * x[v, r] * (g1[v, r] + g2[r] + min_s_dist[r]) for r in pickup_node_ids
            ) <= max(vehicles.battery_level[idle_vehicles[v]], 0)
             for v in AET_HET),
            name='constraint(4)'
        )

        model.optimize()

        if model.status == GRB.OPTIMAL:
            vehicles.controller.profits += model.objVal
            for v in vehicle_ids:
                for r in pickup_node_ids:
                    if x[v, r].X > 0:
                        if v in AET:
                            c = c3
                        elif v in HET:
                            c = c4
                        elif v in HGT:
                            c = c5
                        else:
                            raise
                        vehicles.controller.profits += c * (g1[v, r] + g2[r])
                        ped_idx = peds_to_assign[r]
                        ped_id = pedestrians.id[ped_idx]
                        veh_idx = idle_vehicles[v]
                        veh_id = vehicles.id[veh_idx]
                        # for debug
                        try:
                            if veh_idx in moving_to_pl_AET:
                                network.shortest_path(vehicles.next_node[veh_idx], pedestrians.pickup_node_id[ped_idx])
                            else:
                                network.shortest_path(vehicles.current_node[veh_idx], pedestrians.pickup_node_id[ped_idx])
                        except:
                            vehicles.controller.profits -= (c1 + c2 * g2[r])
                            continue
                        vehicles.controller.assigned_peds_num[veh_idx] += 1
                        pedestrians.state_container.change_agent_state(np.array([ped_id], dtype=int),
                                                                       PedestrianStates.Assigned)
                        # for debug
                        vehicles.order_sequence[veh_idx].append((ped_id, pedestrians.orig_lon_deg[ped_idx],
                                                                 pedestrians.orig_lat_deg[ped_idx],
                                                                 int(pedestrians.pickup_node_id[ped_idx]),
                                                                 vehicles.current_node[veh_idx]))

                        # pickup
                        task1 = vehicles.task_container.generate_task(
                            VehicleTasks.TravelToPickUp,
                            veh_id,
                            "customer_ids", ped_id,
                            "destination_long", pedestrians.orig_lon_deg[ped_idx],
                            "destination_lat", pedestrians.orig_lat_deg[ped_idx],
                            "destination_node", int(pedestrians.pickup_node_id[ped_idx]),
                        )
                        task2 = vehicles.task_container.generate_task(
                            VehicleTasks.PickupCustomer,
                            veh_id,
                            "customer_ids", ped_id,
                            "duration", pedestrians.pickup_duration[ped_idx],
                        )
                        vehicles.add_tasks_by_id(veh_id, task1)
                        vehicles.add_tasks_by_id(veh_id, task2)

                        # dropoff
                        task3 = vehicles.task_container.generate_task(
                            VehicleTasks.TravelToDropOff,
                            veh_id,
                            "customer_ids", ped_id,
                            "destination_long", pedestrians.dest_lon_deg[ped_idx],
                            "destination_lat", pedestrians.dest_lat_deg[ped_idx],
                            "destination_node", int(pedestrians.dropoff_node_id[ped_idx]),
                        )
                        task4 = vehicles.task_container.generate_task(
                            VehicleTasks.DropoffCustomer,
                            veh_id,
                            "customer_ids", ped_id,
                            "duration", pedestrians.dropoff_duration[ped_idx],
                        )
                        vehicles.add_tasks_by_id(veh_id, task3)
                        vehicles.add_tasks_by_id(veh_id, task4)

        # Send AETs with low battery levels to be charged
        min_battery_threshold, max_battery_threshold = 0.2, 0.8
        for i in idle_AET:
            veh_idx = idle_vehicles[i]
            if vehicles.task_queue[veh_idx]:
                continue
            veh_id = vehicles.id[veh_idx]
            if vehicles.battery_level[veh_idx] < vehicles.battery_capacity[veh_idx] * min_battery_threshold:
                dist_to_pl = np.zeros(pl_num)
                for pl_idx in range(pl_num):
                    try:
                        dist_to_pl[pl_idx] = network.shortest_path_cost(vehicles.current_node[veh_idx],
                                                                        parkinglots.node_id[pl_idx])
                    except KeyError:
                        dist_to_pl[pl_idx] = 100000
                dist_to_pl = m_to_km(dist_to_pl)
                pl_idx = np.argmin(dist_to_pl)
                task1 = vehicles.task_container.generate_task(
                    VehicleTasks.TravelToParkingLot,
                    veh_id,
                    "destination_long", parkinglots.lon_deg[pl_idx],
                    "destination_lat", parkinglots.lat_deg[pl_idx],
                    "destination_node", parkinglots.node_id[pl_idx],
                )
                charge_duration = hours_to_seconds((vehicles.battery_capacity[veh_idx] * max_battery_threshold -
                                                    vehicles.battery_level[veh_idx] +
                                                    dist_to_pl[pl_idx] * vehicles.energy_consumption[veh_idx]) /
                                                   vehicles.charge_rate[veh_idx])
                task2 = vehicles.task_container.generate_task(
                    VehicleTasks.Charging,
                    veh_id,
                    "duration", charge_duration
                )
                vehicles.add_tasks_by_id(veh_id, task1)
                vehicles.add_tasks_by_id(veh_id, task2)

    @staticmethod
    def step_assignment_simple(pedestrians, vehicles, parkinglots,
                               network: Network, current_time, maxRuntime=60):
        speed = vehicles.spd_kmh[0]

        peds_to_assign = pedestrians.index_by_state(PedestrianStates.Unassigned)
        orders_num = len(peds_to_assign)
        if orders_num == 0:
            return
        pickup_node_ids = np.arange(0, orders_num)
        pl_num = len(parkinglots.id)
        parkinglot_ids = np.arange(0, pl_num)

        # Nearest matching: AETs' Charging/Queuing/TravelingToParkinglot can't be terminated
        # online vehicles = idle vehicles
        idle_vehicles_origin = vehicles.index_by_state(VehicleStates.Idle)
        vehicle_num = len(idle_vehicles_origin)
        if vehicle_num == 0:
            return

        # There are four types of vehicles:
        # 1.AET: Autonomous, Electric 2.AGT: Autonomous, Gasoline 3.HET: Human-driven, Electric 4.Human-driven, Gasoline
        AET, AGT, HET, HGT = [], [], [], []
        idle_vehicles = []
        i, j = 0, 0
        while i < vehicle_num:
            veh_idx = idle_vehicles_origin[j]
            veh_id = vehicles.id[veh_idx]
            if vehicles.is_electric[veh_idx]:
                if vehicles.is_autonomous[veh_idx]:
                    idle_vehicles.append(veh_idx)
                    AET.append(i)
                    i += 1
                else:
                    min_battery_threshold = 0.75 - vehicles.range_anxiety[veh_idx] * 0.05
                    if vehicles.battery_level[veh_idx] < vehicles.battery_capacity[veh_idx]*min_battery_threshold:
                        # offline to charge
                        vehicle_num -= 1
                        vehicles.is_online[veh_idx] = False
                        dist_to_pl = np.zeros(pl_num)
                        for pl_idx in range(pl_num):
                            dist_to_pl[pl_idx] = network.shortest_path_cost(vehicles.current_node[veh_idx], parkinglots.node_id[pl_idx])
                        dist_to_pl = m_to_km(dist_to_pl)
                        pl_idx = np.argmin(dist_to_pl)
                        task1 = vehicles.task_container.generate_task(
                            VehicleTasks.OfflineToParkingLot,
                            veh_id,
                            "destination_long", parkinglots.lon_deg[pl_idx],
                            "destination_lat", parkinglots.lat_deg[pl_idx],
                            "destination_node", parkinglots.node_id[pl_idx],
                        )
                        max_battery_threshold = np.random.normal(0.85, 0.3)
                        while ((max_battery_threshold < min_battery_threshold) or (max_battery_threshold > 1) or
                               (max_battery_threshold < 0.7)):
                            max_battery_threshold = np.random.normal(0.85, 0.3)
                        charge_duration = hours_to_seconds(((vehicles.battery_capacity[veh_idx] * max_battery_threshold
                                                             - vehicles.battery_level[veh_idx] +
                                                             dist_to_pl[pl_idx] * vehicles.energy_consumption[veh_idx])
                                                            / vehicles.charge_rate[veh_idx]))
                        task2 = vehicles.task_container.generate_task(
                            VehicleTasks.OfflineCharging,
                            veh_id,
                            "duration", charge_duration
                        )
                        vehicles.add_tasks_by_id(veh_id, task1)
                        vehicles.add_tasks_by_id(veh_id, task2)
                        j += 1
                        continue
                    elif vehicles.cont_op_time[veh_idx] > hours_to_seconds(vehicles.max_cont_op_time[veh_idx]):
                        # finished work
                        vehicle_num -= 1
                        vehicles.is_online[veh_idx] = False
                        vehicles.state_container.change_agent_state([veh_id], VehicleStates.Finished)
                        j += 1
                        continue
                    else:
                        idle_vehicles.append(veh_idx)
                        HET.append(i)
                        i += 1
            else:
                if vehicles.is_autonomous[veh_idx]:
                    idle_vehicles.append(veh_idx)
                    AGT.append(i)
                    i += 1
                else:
                    if vehicles.cont_op_time[veh_idx] > hours_to_seconds(vehicles.max_cont_op_time[veh_idx]):
                        # finished work
                        vehicle_num -= 1
                        vehicles.is_online[veh_idx] = False
                        vehicles.state_container.change_agent_state([veh_id], VehicleStates.Finished)
                        j += 1
                        continue
                    else:
                        idle_vehicles.append(veh_idx)
                        HGT.append(i)
                        i += 1
            j += 1
        if vehicle_num == 0:
            return
        idle_vehicles = np.array(idle_vehicles)
        vehicle_ids = np.arange(0, vehicle_num)

        AGT_HGT = AGT + HGT
        AET_HET = AET + HET
        if len(AET_HET) > 0:
            u1 = vehicles.energy_consumption[idle_vehicles[AET_HET[0]]]
        else:
            u1 = 0

        # distance matrix
        # TODO: Use param to get the osrm_url?
        osrm_url = "http://localhost:{}/table/v1/driving/".format(network.owner.params['osrm_port'])
        # g1: vehicle2pickup g2: pickup2dropoff(no need to use matrix?) g3: dropoff2parkinglot
        coordinates1, coordinates2, coordinates3 = "", "", ""

        # TODO: Use get_location to get the coord of nodes?
        sources1, destinations1 = "", ""
        for v in vehicle_ids:
            sources1 = sources1 + ";" + str(v)
            veh_idx = idle_vehicles[v]
            coordinates1 = (coordinates1 + ";" +
                            str(network.graph.nodes[vehicles.current_node[veh_idx]]['x']) + "," +
                            str(network.graph.nodes[vehicles.current_node[veh_idx]]['y']))
        for o in pickup_node_ids:
            destinations1 = destinations1 + ";" + str(vehicle_num + o)
            coordinates1 = (coordinates1 + ";" +
                            str(network.graph.nodes[pedestrians.pickup_node_id[peds_to_assign[o]]]['x']) + ","
                            + str(network.graph.nodes[pedestrians.pickup_node_id[peds_to_assign[o]]]['y']))
        sources1 = sources1[1:]
        destinations1 = destinations1[1:]
        coordinates1 = coordinates1[1:]
        request_url = f"{osrm_url}{coordinates1}?sources={sources1}&destinations={destinations1}&annotations=distance"
        response = requests.get(request_url)
        if response.status_code == 200:
            data = response.json()
            g1 = np.array(data['distances'])
            g1[g1 == None] = 999999
            g1 = m_to_km(g1)
        else:
            raise Exception(
                f"Error fetching data from OSRM for O-D pair: {response.status_code}")

        sources2, destinations2 = "", ""
        for o in pickup_node_ids:
            sources2 = sources2 + ";" + str(o)
            coordinates2 = (coordinates2 + ";" +
                            str(network.graph.nodes[pedestrians.pickup_node_id[peds_to_assign[o]]]['x']) + "," +
                            str(network.graph.nodes[pedestrians.pickup_node_id[peds_to_assign[o]]]['y']))
        for o in pickup_node_ids:
            destinations2 = destinations2 + ";" + str(orders_num + o)
            coordinates2 = (coordinates2 + ";" +
                            str(network.graph.nodes[pedestrians.dropoff_node_id[peds_to_assign[o]]]['x']) + "," +
                            str(network.graph.nodes[pedestrians.dropoff_node_id[peds_to_assign[o]]]['y']))
        sources2 = sources2[1:]
        destinations2 = destinations2[1:]
        coordinates2 = coordinates2[1:]
        request_url = f"{osrm_url}{coordinates2}?sources={sources2}&destinations={destinations2}&annotations=distance"
        response = requests.get(request_url)
        if response.status_code == 200:
            data = response.json()
            g2 = np.diag(np.array(data['distances']))
            g2 = m_to_km(g2)
        else:
            raise Exception(
                f"Error fetching data from OSRM for O-D pair: {response.status_code}")

        sources3, destinations3 = "", ""
        for d in pickup_node_ids:
            sources3 = sources3 + ';' + str(d)
            coordinates3 = (coordinates3 + ";" +
                            str(network.graph.nodes[pedestrians.dropoff_node_id[peds_to_assign[d]]]['x']) + "," +
                            str(network.graph.nodes[pedestrians.dropoff_node_id[peds_to_assign[d]]]['y']))
        for s in parkinglot_ids:
            destinations3 = destinations3 + ";" + str(orders_num + s)
            coordinates3 = (coordinates3 + ";" +
                            str(network.graph.nodes[parkinglots.node_id[s]]['x']) + "," +
                            str(network.graph.nodes[parkinglots.node_id[s]]['y']))
        sources3 = sources3[1:]
        destinations3 = destinations3[1:]
        coordinates3 = coordinates3[1:]
        request_url = f"{osrm_url}{coordinates3}?sources={sources3}&destinations={destinations3}&annotations=distance"
        response = requests.get(request_url)
        if response.status_code == 200:
            data = response.json()
            g3 = np.array(data['distances'])
            g3[g3 == None] = 999999
            g3 = m_to_km(g3)
        else:
            raise Exception(
                f"Error fetching data from OSRM for O-D pair: {response.status_code}")
        min_s_dist = {r: min(g3[r, s] for s in parkinglot_ids) for r in pickup_node_ids}

        # time matrix
        h1 = hours_to_seconds(g1/speed)

        c1 = 14
        c2 = 7
        c3 = 4.031
        c4 = 4.976
        c5 = 6.176
        c6 = 7.121

        assigned_vehicles = []
        for r in pickup_node_ids:
            dist_from_vehicles = g1[:, r]
            sorted_idx = np.argsort(dist_from_vehicles)
            for i in range(vehicle_num):
                v = sorted_idx[i]
                if v in assigned_vehicles:
                    continue
                if v in AET_HET:
                    energy_cost = (g1[v, r] + g2[r] + min_s_dist[r]) * u1
                    if energy_cost >= max(vehicles.battery_level[idle_vehicles[v]], 0):
                        continue
                if current_time + h1[v, r] >= pedestrians.pickup_tw_end[peds_to_assign[r]]:
                    continue
                ped_idx = peds_to_assign[r]
                ped_id = pedestrians.id[ped_idx]
                veh_idx = idle_vehicles[v]
                veh_id = vehicles.id[veh_idx]
                try:
                    network.shortest_path(vehicles.current_node[veh_idx], pedestrians.pickup_node_id[ped_idx])
                except:
                    continue
                pedestrians.state_container.change_agent_state(np.array([ped_id], dtype=int),
                                                               PedestrianStates.Assigned)
                # for debug
                vehicles.order_sequence[veh_idx].append((ped_id, pedestrians.orig_lon_deg[ped_idx],
                                                         pedestrians.orig_lat_deg[ped_idx],
                                                         int(pedestrians.pickup_node_id[ped_idx]),
                                                         vehicles.current_node[veh_idx]))

                if v in AET:
                    c = c3
                elif v in AGT:
                    c = c4
                elif v in HET:
                    c = c5
                elif v in HGT:
                    c = c6
                else:
                    raise
                vehicles.controller.profits += c1 + c2 * g2[r] - c * (g1[v, r] + g2[r])
                assigned_vehicles.append(v)
                # pickup
                task1 = vehicles.task_container.generate_task(
                    VehicleTasks.TravelToPickUp,
                    veh_id,
                    "customer_ids", ped_id,
                    "destination_long", pedestrians.orig_lon_deg[ped_idx],
                    "destination_lat", pedestrians.orig_lat_deg[ped_idx],
                    "destination_node", int(pedestrians.pickup_node_id[ped_idx]),
                )
                task2 = vehicles.task_container.generate_task(
                    VehicleTasks.PickupCustomer,
                    veh_id,
                    "customer_ids", ped_id,
                    "duration", pedestrians.pickup_duration[ped_idx],
                )
                vehicles.add_tasks_by_id(veh_id, task1)
                vehicles.add_tasks_by_id(veh_id, task2)

                # dropoff
                task3 = vehicles.task_container.generate_task(
                    VehicleTasks.TravelToDropOff,
                    veh_id,
                    "customer_ids", ped_id,
                    "destination_long", pedestrians.dest_lon_deg[ped_idx],
                    "destination_lat", pedestrians.dest_lat_deg[ped_idx],
                    "destination_node", int(pedestrians.dropoff_node_id[ped_idx]),
                )
                task4 = vehicles.task_container.generate_task(
                    VehicleTasks.DropoffCustomer,
                    veh_id,
                    "customer_ids", ped_id,
                    "duration", pedestrians.dropoff_duration[ped_idx],
                )
                vehicles.add_tasks_by_id(veh_id, task3)
                vehicles.add_tasks_by_id(veh_id, task4)
                break

        # Send AETs with low battery levels to be charged
        min_battery_threshold, max_battery_threshold = 0.2, 0.8
        for i in AET:
            veh_idx = idle_vehicles[i]
            if vehicles.task_queue[veh_idx]:
                continue
            veh_id = vehicles.id[veh_idx]
            if vehicles.battery_level[veh_idx] < vehicles.battery_capacity[veh_idx] * min_battery_threshold:
                dist_to_pl = np.zeros(pl_num)
                for pl_idx in range(pl_num):
                    dist_to_pl[pl_idx] = network.shortest_path_cost(vehicles.current_node[veh_idx], parkinglots.node_id[pl_idx])
                dist_to_pl = m_to_km(dist_to_pl)
                pl_idx = np.argmin(dist_to_pl)
                task1 = vehicles.task_container.generate_task(
                    VehicleTasks.TravelToParkingLot,
                    veh_id,
                    "destination_long", parkinglots.lon_deg[pl_idx],
                    "destination_lat", parkinglots.lat_deg[pl_idx],
                    "destination_node", parkinglots.node_id[pl_idx],
                )
                charge_duration = hours_to_seconds((vehicles.battery_capacity[veh_idx] * max_battery_threshold -
                                                    vehicles.battery_level[veh_idx] +
                                                    dist_to_pl[pl_idx] * vehicles.energy_consumption[veh_idx]) /
                                                   vehicles.charge_rate[veh_idx])
                task2 = vehicles.task_container.generate_task(
                    VehicleTasks.Charging,
                    veh_id,
                    "duration", charge_duration
                )
                vehicles.add_tasks_by_id(veh_id, task1)
                vehicles.add_tasks_by_id(veh_id, task2)
