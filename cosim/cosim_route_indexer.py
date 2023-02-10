from collections import OrderedDict
from dictor import dictor

import copy
import random

from srunner.scenarioconfigs.route_scenario_configuration import RouteScenarioConfiguration

from leaderboard.utils.route_indexer import RouteIndexer
from leaderboard.utils.route_parser import RouteParser
from leaderboard.utils.checkpoint_tools import fetch_dict, create_default_json_msg, save_dict


class CosimRouteIndexer(RouteIndexer):
    def __init__(self, routes_file, scenarios_file, repetitions, shuffle=False):
        super().__init__(routes_file, scenarios_file, repetitions)

        if shuffle: 
            random.shuffle(self._configs_list)


    def set_route(self, index):
        if index >= len(self._configs_list):
            return None

        key, config = self._configs_list[index]
        self._index = index

        return config

