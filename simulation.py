from simple_playground import Playground, PSO, MLP  

def simulate_car_runs(num_runs=500):
    playground = Playground()
    pso = PSO(init_particleNumber=50)
    pso.train_model()

    success_count = 0
    for _ in range(num_runs):
        playground.reset()
        while not playground.done:
            state = playground.state
            wheel_angle = MLP(state, pso.get_weight()[:-1], pso.get_weight()[-1].reshape(-1, 1))
            playground.step(wheel_angle)
        if playground.complete:
            success_count += 1

    success_probability = success_count / num_runs
    return success_probability

if __name__ == '__main__':
    success_probability = simulate_car_runs()
    print(f"成功率: {success_probability:.2%}")
