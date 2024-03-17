import os
from math import copysign


class Vehicle:
    def __init__(self, presence, x, y, vx, vy, cos_h, sin_h):
        self.presence = presence
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.sin_h = sin_h
        self.cos_h = cos_h

class RoadObject:
    def __init__(self,presence, x,y):
        self.presence = presence
        self.x = x
        self.y = y



class SafetyRulesParser:

    class ParsedState:
        def __init__(self, ego):
            self.ego = ego
            self.others = []
            self.objects = []
            self.is_right = 1
            self.is_left = 1

        def addOtherVehicle(self, other):
            self.others.append(other)

        def addObject(self, obj):
            self.objects.append(obj)

    def __init__(self, rules_path, actions):
        self.rules = {"VEHICLE": {}, "OBJECT": {}}
        self.rules_path = rules_path
        self.open_time = 0
        self.parse_rules()
        self.actions = actions


    def check_state_action_if_safe(self,state, action):
        iters = int(len(state-10)/7 - 2)
        parsed = self.ParsedState(Vehicle(state[0].item(),state[1].item(),state[2].item(),state[3].item(),state[4].item(),state[5].item(),state[6].item()))
        for iter in range(iters):
            start_index = (iter+1)*7
            parsed.addOtherVehicle(Vehicle(state[start_index].item(), state[start_index + 1].item(), state[start_index + 2].item(), state[start_index + 3].item(), state[start_index + 4].item(), state[start_index + 5].item(), state[start_index + 6].item()))
        parsed.addObject(RoadObject(state[start_index+7],state[start_index+8],state[start_index+9]))
        parsed.is_left = state[start_index+10]
        parsed.is_right = state[start_index + 11]
        action_name = self.actions[action]
        vehicle_action_rules = self.rules['VEHICLE'].get(action_name, [])
        object_action_rules = self.rules['OBJECT'].get(action_name, [])
        ego = parsed.ego
        for rule in vehicle_action_rules:
            for v in parsed.others:
                if eval(rule):
                    return False
        for rule in object_action_rules:
            for o in parsed.objects:
                if eval(rule):
                    return False
        return True


    def parse_rules(self):
        curr_time = os.path.getmtime(self.rules_path)
        if self.open_time == curr_time:
            return
        self.open_time = curr_time
        with open(self.rules_path) as f:
            lines = f.readlines()
            for line in lines:
                rule = line.split(sep=';')
                if rule[1] not in self.rules[rule[0]]:
                    self.rules[rule[0]][rule[1]] = []
                self.rules[rule[0]][rule[1]].append(rule[2].rstrip())

