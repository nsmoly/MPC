
# (c) 2022 Nikolai Smolyanskiy
# This code is licensed under MIT license (see LICENSE.txt). No warranties

# This code demonstrates Model Predictive Control applied to
# kinematic bicycle model that can be used for car control. 
# This project uses PyTorch for optimization. See README.md for details

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# Kinematic bicycle model. Implements forward pass that creates full trajectory given controls
# this model can be optimized via torch.optim optimizers just like any NN
class BicycleModel(nn.Module):
    def __init__(self, device, n_steps=10, wheelbase=2.7, dt=0.05, max_steer=30.0, friction_road=0.1, friction_air=0.01):
        super().__init__()
        self.device = device

        # All parameters are in metric system: m, s, m/s, radians...
        self.n_steps = int(n_steps)
        self.wheelbase = torch.tensor(wheelbase).to(device)
        self.max_steer = torch.tensor(np.deg2rad(max_steer, dtype=float)).to(device)
        self.max_speed = torch.tensor(100.0).to(device)
        self.dt = torch.tensor(dt).to(device)
        self.friction_road = torch.tensor(friction_road).to(device)
        self.friction_air = torch.tensor(friction_air).to(device)
        
        # State
        self.x = [torch.tensor(0.0).to(device) for i in range(0, self.n_steps)]
        self.y = [torch.tensor(0.0).to(device) for i in range(0, self.n_steps)]
        self.yaw = [torch.tensor(0.0).to(device) for i in range(0, self.n_steps)]
        self.speed = [torch.tensor(0.0).to(device) for i in range(0, self.n_steps)]

        # Control - accelerations and steering angles (front wheels angle) for all time steps
        self.accel = nn.ParameterList([nn.Parameter(torch.tensor(0.0, requires_grad=True)).to(device) for i in range(0, self.n_steps)])
        self.steering = nn.ParameterList([nn.Parameter(torch.tensor(0.0, requires_grad=True)).to(device) for i in range(0, self.n_steps)])

    def set_controls(self, accelerations, steering_angles):
        self.accel = nn.ParameterList([nn.Parameter(torch.tensor(accelerations[i], requires_grad=True).to(self.device)) for i in range(0, self.n_steps)])
        self.steering = nn.ParameterList([nn.Parameter(torch.tensor(steering_angles[i], requires_grad=True).to(self.device)) for i in range(0, self.n_steps)])

    def get_controls(self):
        accel_list = [self.accel[i].item() for i in range(0, self.n_steps)]
        steering_list = [self.steering[i].item() for i in range(0, self.n_steps)]
        return accel_list, steering_list

    def forward(self, start_x, start_y, start_yaw, start_speed):
        # Set initial conditions
        self.x[0] = start_x
        self.y[0] = start_y
        self.yaw[0] = start_yaw
        self.speed[0] = start_speed

        for i in range(0, self.n_steps-1):
            # Compute speed
            friction = self.speed[i]*self.friction_road + self.friction_air*self.speed[i]*self.speed[i]
            self.speed[i+1] = torch.clamp(self.speed[i] + self.dt*(self.accel[i] - friction), 0, self.max_speed)
            
            # Clamp steering control and compute current angular velocity
            steering_angle = torch.clamp(self.steering[i], -self.max_steer, self.max_steer)
            angular_velocity = self.speed[i]*torch.tan(steering_angle)/self.wheelbase

            self.x[i+1] = self.x[i] + self.speed[i]*torch.cos(self.yaw[i])*self.dt
            self.y[i+1] = self.y[i] + self.speed[i]*torch.sin(self.yaw[i])*self.dt
            self.yaw[i+1] = self.yaw[i] + angular_velocity*self.dt
        
        return self.x, self.y, self.yaw, self.speed


def optimize_params(device, model, step_count, ref_x, ref_y, start_yaw, start_speed):
    w_mse = 1.0
    w_reg = 0.5
    w_target_pos = 0.0

    accels = np.zeros(step_count, dtype=float).tolist()
    steering = np.zeros(step_count, dtype=float).tolist()
    model.set_controls(accels, steering)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.1, amsgrad=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    target_x = [torch.tensor(ref_x[i]).to(device) for i in range(0, len(ref_x))]
    target_y = [torch.tensor(ref_y[i]).to(device) for i in range(0, len(ref_y))]
    x_0 = torch.tensor(ref_x[0]).to(device)
    y_0 = torch.tensor(ref_y[0]).to(device)
    yaw_0 = torch.tensor(start_yaw).to(device)
    speed_0 = torch.tensor(start_speed).to(device)

    losses_log = []
    for iter in range(0, 100):
        optimizer.zero_grad()
        
        x, y, yaw, speed = model(x_0, y_0, yaw_0, speed_0)
        errors = [(x[i]-target_x[i])**2.0 + (y[i]-target_y[i])**2.0 for i in range(0, len(x))]
        constraints = [parameter**2.0 for parameter in model.parameters()]

        loss_mse = torch.sum(torch.stack(errors, dim=0), dim=0)
        loss_target_position = (x[-1]-target_x[-1])**2.0 + (y[-1]-target_y[-1])**2.0
        loss_reg = torch.sum(torch.stack(constraints, dim=0), dim=0)
        total_loss = w_mse*loss_mse + w_target_pos*loss_target_position + w_reg*loss_reg
        #print(f"Loss: {loss_mse.item()}")
        total_loss.backward()
        optimizer.step()
        #scheduler.step()

        losses_log.append(loss_mse.item())

    accels, steering = model.get_controls()

    return accels, steering, losses_log


def main():
    device = torch.device("cpu") 
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    control_freq = 5.0
    time_horizon = 5.0
    delta_t = 1.0/control_freq
    step_count = int(control_freq*time_horizon)

    print(f"Running bicycle model for {step_count} steps with timestep {delta_t}")

    model = BicycleModel(device=device, n_steps=step_count, dt=delta_t)
    print(f"Computing on: {next(model.parameters()).device}")

    accels = [0.0 for i in range(0, step_count)]
    steering = [0.0 for i in range(0, step_count)]
    for i in range(0, int(step_count/2.0)):
        accels[i] = 3.0
    for i in range(0, int(step_count/2.0)):
        steering[i] = np.deg2rad(20.0, dtype=float)
    for i in range(int(step_count/2.0), step_count):
        steering[i] = np.deg2rad(-25.0, dtype=float)
    
    model.set_controls(accels, steering)

    x_0 = torch.tensor(0.0).to(device)
    y_0 = torch.tensor(0.0).to(device)
    yaw_0 = torch.tensor(0.0).to(device)
    speed_0 = torch.tensor(0.0).to(device)
    x, y, yaw, speed = model(x_0, y_0, yaw_0, speed_0)
    print("Initial setup:")
    print(f"accel: {accels}")
    print(f"steering: {steering}")
    print(f"x: {[x[i].item() for i in range(0, step_count)]}")
    print(f"y: {[y[i].item() for i in range(0, step_count)]}")
    print(f"yaw: {[yaw[i].item() for i in range(0, step_count)]}")
    print(f"speed: {[speed[i].item() for i in range(0, step_count)]}")

    plt.axes().set_aspect('equal', 'datalim')
    plt.plot(
        torch.stack(x).detach().cpu().numpy(), torch.stack(y).detach().cpu().numpy(), 'bo',
        torch.stack(x).detach().cpu().numpy(), torch.stack(y).detach().cpu().numpy(), 'k')
    plt.show()

    ref_x = [x[i].item() for i in range(0, len(x))]
    ref_y = [y[i].item() for i in range(0, len(y))]

    print()
    print("Optimizing controls:")
    time_start = time.time()
    accels, steering, losses_log = optimize_params(device, model, step_count, ref_x, ref_y, 0.0, 0.0)
    print(f"Runtime: {(time.time() - time_start)} s")
    print(f"Start loss={losses_log[0]}")
    print(f"Final loss={losses_log[-1]}")

    # Show results
    plt.plot(losses_log, 'r')
    plt.show()

    x_0 = torch.tensor(0.0).to(device)
    y_0 = torch.tensor(0.0).to(device)
    yaw_0 = torch.tensor(0.0).to(device)
    speed_0 = torch.tensor(0.0).to(device)
    x, y, yaw, speed = model(x_0, y_0, yaw_0, speed_0)
    plt.axes().set_aspect('equal', 'datalim')
    plt.plot(
        torch.stack(x).detach().cpu().numpy(), torch.stack(y).detach().cpu().numpy(), 'ro',
        torch.stack(x).detach().cpu().numpy(), torch.stack(y).detach().cpu().numpy(), 'k')
    plt.plot(ref_x, ref_y, 'bo', ref_x, ref_y, 'k')
    plt.show()
    
    print()
    print("Optimizated parameters:")
    print(f"accel: {accels}")
    print(f"steering: {steering}")

    print(f"Ran on {next(model.parameters()).device}")
    print("Done!")


if __name__ == "__main__":
    main()