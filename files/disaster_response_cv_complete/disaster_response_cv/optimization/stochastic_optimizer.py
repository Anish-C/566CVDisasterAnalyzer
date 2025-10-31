"""
Two-Stage Stochastic Optimization for Disaster Response Resource Allocation

Stage 1: Initial resource deployment (before recourse)
Stage 2: Recourse adjustments based on realized scenario

Objective: Minimize expected casualties + deployment costs
"""

import numpy as np
from pyomo.environ import *
import time


class DisasterResponseOptimizer:
    """
    Two-stage stochastic programming model for disaster response
    
    Decision variables:
    - x[i,r]: Stage 1 - Initial deployment of resource type r to location i
    - y[i,r,s]: Stage 2 - Recourse allocation in scenario s
    
    Objective: Minimize expected casualties + costs
    """
    
    def __init__(self, n_buildings, n_resource_types=5, solver='glpk'):
        """
        Args:
            n_buildings: Number of buildings to allocate resources to
            n_resource_types: Number of different resource types
            solver: Optimization solver to use ('glpk', 'gurobi', 'cplex')
        """
        self.n_buildings = n_buildings
        self.n_resource_types = n_resource_types
        self.solver = solver
        
        # Resource types: 0=Search/Rescue, 1=Medical, 2=Heavy Equipment, 3=Supplies, 4=Flexible
        self.resource_names = ['Search_Rescue', 'Medical', 'Heavy_Equipment', 'Supplies', 'Flexible']
        
    def solve(self, scenarios, probabilities, building_info, resource_constraints):
        """
        Solve two-stage stochastic optimization problem
        
        Args:
            scenarios: Array of shape (n_scenarios, n_buildings) - damage state per scenario
            probabilities: Array of shape (n_scenarios,) - scenario probabilities
            building_info: Dict with building characteristics (population, location, etc.)
            resource_constraints: Dict with resource availability and costs
            
        Returns:
            solution: Dict with optimal allocation and objective value
        """
        print(f"\n{'='*60}")
        print(f"SOLVING TWO-STAGE STOCHASTIC OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Buildings: {self.n_buildings}")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Resource types: {self.n_resource_types}")
        
        start_time = time.time()
        
        # Create model
        model = ConcreteModel()
        
        # Sets
        model.Buildings = RangeSet(0, self.n_buildings - 1)
        model.Resources = RangeSet(0, self.n_resource_types - 1)
        model.Scenarios = RangeSet(0, len(scenarios) - 1)
        
        # Parameters
        model.scenario_prob = Param(model.Scenarios, initialize=dict(enumerate(probabilities)))
        model.population = Param(model.Buildings, initialize=building_info['population'])
        
        # Casualty rates by damage state (0=no damage, 1=minor, 2=major, 3=destroyed)
        casualty_rates = {0: 0.0, 1: 0.05, 2: 0.20, 3: 0.50}
        
        # Initialize damage states per scenario
        damage_dict = {}
        for s in range(len(scenarios)):
            for i in range(self.n_buildings):
                damage_dict[(i, s)] = int(scenarios[s, i])
        model.damage = Param(model.Buildings, model.Scenarios, initialize=damage_dict)
        
        # Resource costs and availability
        model.resource_cost = Param(model.Resources, initialize=resource_constraints['costs'])
        model.resource_capacity = Param(model.Resources, initialize=resource_constraints['capacity'])
        
        # Resource effectiveness by damage type
        # effectiveness[resource][damage_state] = reduction in casualty rate
        effectiveness = self._get_resource_effectiveness()
        
        # Decision Variables
        # Stage 1: Initial deployment
        model.x = Var(model.Buildings, model.Resources, domain=NonNegativeReals, bounds=(0, 10))
        
        # Stage 2: Recourse adjustments
        model.y = Var(model.Buildings, model.Resources, model.Scenarios, 
                     domain=NonNegativeReals, bounds=(0, 10))
        
        # Auxiliary: Total resources at each building in each scenario
        model.total_resources = Var(model.Buildings, model.Resources, model.Scenarios,
                                   domain=NonNegativeReals)
        
        # Define total resources constraint
        def total_resources_rule(model, i, r, s):
            return model.total_resources[i, r, s] == model.x[i, r] + model.y[i, r, s]
        model.total_resources_constraint = Constraint(
            model.Buildings, model.Resources, model.Scenarios, rule=total_resources_rule
        )
        
        # Objective Function: Minimize expected casualties + costs
        def objective_rule(model):
            # Stage 1 costs
            stage1_cost = sum(model.resource_cost[r] * sum(model.x[i, r] 
                            for i in model.Buildings) 
                            for r in model.Resources)
            
            # Stage 2: Expected casualties and recourse costs
            expected_casualties = 0
            expected_recourse_cost = 0
            
            for s in model.Scenarios:
                scenario_casualties = 0
                scenario_recourse = 0
                
                for i in model.Buildings:
                    damage = int(scenarios[s, i])
                    base_casualty_rate = casualty_rates[damage]
                    
                    # Calculate reduction from resources
                    reduction = 0
                    for r in model.Resources:
                        # Effectiveness depends on resource type and damage state
                        eff = effectiveness[r][damage]
                        reduction += eff * model.total_resources[i, r, s]
                    
                    # Casualty rate reduced by resources (but can't go below 0)
                    effective_rate = max(0, base_casualty_rate - reduction)
                    scenario_casualties += effective_rate * model.population[i]
                    
                    # Recourse costs
                    for r in model.Resources:
                        scenario_recourse += 1.5 * model.resource_cost[r] * model.y[i, r, s]
                
                expected_casualties += model.scenario_prob[s] * scenario_casualties
                expected_recourse_cost += model.scenario_prob[s] * scenario_recourse
            
            # Weight casualties heavily (convert to cost equivalent)
            casualty_cost_weight = 1000  # Each casualty costs $1000 equivalent
            
            return stage1_cost + expected_casualties * casualty_cost_weight + expected_recourse_cost
        
        model.obj = Objective(rule=objective_rule, sense=minimize)
        
        # Constraints
        
        # 1. Resource availability (Stage 1)
        def resource_availability_rule(model, r):
            return sum(model.x[i, r] for i in model.Buildings) <= model.resource_capacity[r]
        model.resource_availability = Constraint(model.Resources, rule=resource_availability_rule)
        
        # 2. Limited recourse capacity (Stage 2)
        def recourse_limit_rule(model, r, s):
            return sum(model.y[i, r, s] for i in model.Buildings) <= 0.3 * model.resource_capacity[r]
        model.recourse_limit = Constraint(model.Resources, model.Scenarios, rule=recourse_limit_rule)
        
        # 3. Coverage requirements (each high-damage building must get some resources)
        def coverage_rule(model, i, s):
            damage = int(scenarios[s, i])
            if damage >= 2:  # Major or destroyed
                return sum(model.total_resources[i, r, s] for r in model.Resources) >= 1.0
            else:
                return Constraint.Skip
        model.coverage = Constraint(model.Buildings, model.Scenarios, rule=coverage_rule)
        
        # Solve
        print(f"\nSolving with {self.solver}...")
        opt = SolverFactory(self.solver)
        
        if self.solver == 'gurobi':
            opt.options['TimeLimit'] = 300  # 5 minute time limit
            opt.options['MIPGap'] = 0.01  # 1% optimality gap
        
        results = opt.solve(model, tee=False)
        
        solve_time = time.time() - start_time
        
        # Extract solution
        if results.solver.termination_condition == TerminationCondition.optimal:
            print(f"✓ OPTIMAL SOLUTION FOUND in {solve_time:.2f} seconds")
            
            solution = {
                'status': 'optimal',
                'objective_value': value(model.obj),
                'solve_time': solve_time,
                'stage1_allocation': self._extract_stage1(model),
                'expected_casualties': self._compute_expected_casualties(model, scenarios, casualty_rates, effectiveness),
                'stage1_cost': sum(value(model.resource_cost[r] * sum(model.x[i, r] 
                                    for i in model.Buildings)) for r in model.Resources),
                'model': model  # Keep model for detailed analysis
            }
            
            self._print_solution_summary(solution)
            
        else:
            print(f"✗ SOLVER FAILED: {results.solver.termination_condition}")
            solution = {
                'status': 'failed',
                'message': str(results.solver.termination_condition)
            }
        
        return solution
    
    def _get_resource_effectiveness(self):
        """
        Define resource effectiveness by type and damage state
        
        Returns dict: effectiveness[resource_type][damage_state] = reduction rate
        """
        effectiveness = {
            0: {0: 0.00, 1: 0.02, 2: 0.05, 3: 0.08},  # Search/Rescue - most effective for destroyed
            1: {0: 0.00, 1: 0.03, 2: 0.04, 3: 0.03},  # Medical - effective for all
            2: {0: 0.00, 1: 0.01, 2: 0.06, 3: 0.10},  # Heavy Equipment - best for major/destroyed
            3: {0: 0.00, 1: 0.02, 2: 0.03, 3: 0.02},  # Supplies - moderate across all
            4: {0: 0.00, 1: 0.02, 2: 0.04, 3: 0.05},  # Flexible - good all-around
        }
        return effectiveness
    
    def _extract_stage1(self, model):
        """Extract Stage 1 allocation from solved model"""
        allocation = np.zeros((self.n_buildings, self.n_resource_types))
        for i in range(self.n_buildings):
            for r in range(self.n_resource_types):
                allocation[i, r] = value(model.x[i, r])
        return allocation
    
    def _compute_expected_casualties(self, model, scenarios, casualty_rates, effectiveness):
        """Compute expected casualties from solution"""
        expected_casualties = 0
        
        for s in range(len(scenarios)):
            scenario_casualties = 0
            prob = value(model.scenario_prob[s])
            
            for i in range(self.n_buildings):
                damage = int(scenarios[s, i])
                base_rate = casualty_rates[damage]
                
                reduction = 0
                for r in range(self.n_resource_types):
                    eff = effectiveness[r][damage]
                    total_res = value(model.total_resources[i, r, s])
                    reduction += eff * total_res
                
                effective_rate = max(0, base_rate - reduction)
                scenario_casualties += effective_rate * value(model.population[i])
            
            expected_casualties += prob * scenario_casualties
        
        return expected_casualties
    
    def _print_solution_summary(self, solution):
        """Print solution summary"""
        print(f"\n{'='*60}")
        print(f"SOLUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Objective Value: ${solution['objective_value']:,.0f}")
        print(f"Expected Casualties: {solution['expected_casualties']:.1f}")
        print(f"Stage 1 Cost: ${solution['stage1_cost']:,.0f}")
        print(f"Solve Time: {solution['solve_time']:.2f}s")
        
        # Resource allocation summary
        allocation = solution['stage1_allocation']
        print(f"\nStage 1 Resource Allocation:")
        for r in range(self.n_resource_types):
            total = allocation[:, r].sum()
            deployed_to = (allocation[:, r] > 0.01).sum()
            print(f"  {self.resource_names[r]:20s}: {total:6.1f} units → {deployed_to} locations")


if __name__ == "__main__":
    # Test optimization
    print("Testing Two-Stage Stochastic Optimization...")
    
    # Small test case
    n_buildings = 20
    n_scenarios = 10
    
    # Generate test scenarios
    scenarios = np.random.randint(0, 4, size=(n_scenarios, n_buildings))
    probabilities = np.ones(n_scenarios) / n_scenarios
    
    # Building info
    building_info = {
        'population': dict(enumerate(np.random.randint(10, 100, size=n_buildings)))
    }
    
    # Resource constraints
    resource_constraints = {
        'capacity': dict(enumerate([30, 25, 15, 40, 35])),  # Available units per resource type
        'costs': dict(enumerate([100, 150, 300, 50, 120]))  # Cost per unit
    }
    
    # Solve
    optimizer = DisasterResponseOptimizer(n_buildings=n_buildings, solver='glpk')
    solution = optimizer.solve(scenarios, probabilities, building_info, resource_constraints)
    
    if solution['status'] == 'optimal':
        print("\n✓ Test completed successfully!")
    else:
        print(f"\n✗ Test failed: {solution.get('message', 'Unknown error')}")
