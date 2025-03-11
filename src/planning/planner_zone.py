from src.planning.path_planner import PathPlanner

# Search based Planners
from src.planning.search_based.bfs import BFSPlanner
from src.planning.search_based.dfs import DFSPlanner
from src.planning.search_based.best_first import BestFirstPlanner
from src.planning.search_based.a_star import AStarPlanner
from src.planning.search_based.bidirectional_a_star import BidirectionalAStarPlanner
from src.planning.search_based.dijkstra import DijkstraPlanner
from src.planning.search_based.ara_star import ARAStarPlanner
from src.planning.search_based.lpa_star import LPAStarPlanner
from src.planning.search_based.lrta_star import LRTAStarPlanner
from src.planning.search_based.rtaa_star import RTAAStarPlanner
from src.planning.search_based.d_star import DStarPlanner
from src.planning.search_based.d_star_lite import DStarLitePlanner

# Sampling based Planners
from src.planning.sampling_based.rrt import RRTPlanner
from src.planning.sampling_based.dynamic_rrt import DynamicRRTPlanner
from src.planning.sampling_based.rrt_connect import RRTConnectPlanner
from src.planning.sampling_based.extended_rrt import ExtendedRRTPlanner
from src.planning.sampling_based.rrt_star import RRTStarPlanner
from src.planning.sampling_based.informed_rrt_star import InformedRRTStarPlanner
from src.planning.sampling_based.rrt_star_smart import RRTStarSmartPlanner
from src.planning.sampling_based.anytime_rrt_star import AnytimeRRTStarPlanner
from src.planning.sampling_based.closed_loop_rrt_star import ClosedLoopRRTStarPlanner
from src.planning.sampling_based.spline_rrt_star import SplineRRTStarPlanner
from src.planning.sampling_based.bit_star import BITStarPlanner
from src.planning.sampling_based.fmt_star import FMTStarPlanner
from src.planning.sampling_based.gi_rrt import GoalInferenceRRTPlanner

class PlannerZone:
    """
    Factory class for creating path planners.
    
    This class abstracts away the details of planner initialization and
    provides a simple interface for creating different types of planners.
    """
    
    @staticmethod
    def create_planner(planner_type, world, **kwargs):
        """
        Create a planner of the specified type.
        
        Parameters:
            planner_type: String identifying the planner type.
            world: The World object to use for planning.
            **kwargs: Additional parameters specific to the planner.
            
        Returns:
            An initialized planner object.
            
        Raises:
            ValueError: If the planner type is not recognized.
        """
        # Default/Original planner
        if planner_type == "default" or planner_type == "original":
            return PathPlanner(world, **kwargs)
            
        # Search-based planners
        elif planner_type == "bfs":
            return BFSPlanner(world)
        elif planner_type == "dfs":
            return DFSPlanner(world)
        elif planner_type == "best_first":
            return BestFirstPlanner(world)
        elif planner_type == "dijkstra":
            return DijkstraPlanner(world)
        elif planner_type == "a_star":
            return AStarPlanner(world)
        elif planner_type == "bidirectional_a_star":
            return BidirectionalAStarPlanner(world)
        elif planner_type == "ara_star":
            return ARAStarPlanner(world)
        elif planner_type == "lpa_star":
            return LPAStarPlanner(world)
        elif planner_type == "lrta_star":
            return LRTAStarPlanner(world)
        elif planner_type == "rtaa_star":
            return RTAAStarPlanner(world)
        elif planner_type == "d_star_lite":
            return DStarLitePlanner(world)
        elif planner_type == "d_star":
            return DStarPlanner(world)
        
        # Sampling-based planners
        elif planner_type == "rrt":
            return RRTPlanner(world)
        elif planner_type == "dynamic_rrt":
            return DynamicRRTPlanner(world)
        elif planner_type == "rrt_connect":
            return RRTConnectPlanner(world)
        elif planner_type == "extended_rrt":
            return ExtendedRRTPlanner(world)
        elif planner_type == "rrt_star":
            return RRTStarPlanner(world)
        elif planner_type == "anytime_rrt_star":
            return AnytimeRRTStarPlanner(world)
        elif planner_type == "informed_rrt_star":
            return InformedRRTStarPlanner(world)
        elif planner_type == "rrt_star_smart":
            return RRTStarSmartPlanner(world)
        elif planner_type == "closed_loop_rrt_star":
            return ClosedLoopRRTStarPlanner(world)
        elif planner_type == "spline_rrt_star":
            return SplineRRTStarPlanner(world)
        elif planner_type == "bit_star":
            return BITStarPlanner(world)
        elif planner_type == "fmt_star":
            return FMTStarPlanner(world)
        elif planner_type == "gi_rrt":
            return GoalInferenceRRTPlanner(world)

        else:
            raise ValueError(f"Unknown planner type: {planner_type}") 