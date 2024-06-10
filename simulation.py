#測試用
from simple_playground import Playground, PSO, MLP  

def simulate_car_runs(num_runs=100):
    success_count = 0
    for _ in range(num_runs):
        playground = Playground()
        pso = PSO(particle_num=50)
        pso.train_model()

        while not playground.done:
            state = playground.state
            weights = pso.get_weight()
            wheel_angle = MLP(state, weights[:-1], weights[-1].reshape(-1, 1))
            playground.step(wheel_angle)
        
        if playground.complete:
            success_count += 1

    success_probability = success_count / num_runs
    return success_probability

if __name__ == '__main__':
    success_probability = simulate_car_runs()
    print(f"成功率: {success_probability:.2%}")
