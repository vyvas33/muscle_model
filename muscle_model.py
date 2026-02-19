import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

class Muscle:
    def __init__(self, name, F_max_iso, l_opt, l_slack, v_max, N, K, l_ref, w = 0.56, eps_ref = 0.04):
        self.name = name
        self.F_max_iso = F_max_iso  
        self.l_opt = l_opt        
        self.l_slack = l_slack     
        self.v_max = v_max          
        self.N = N                 
        self.K = K 
        self.w = w
        self.eps_ref = eps_ref
        self.l_ref = l_ref

    def force_length_relationship_CE(self, l_ce):
        w = self.w
        l_opt = self.l_opt
        c = np.log(0.05) 

        f_l = np.exp(c*np.abs((l_ce - l_opt) / (l_opt*w))**3)     
        return f_l

    def force_velocity_relationship_CE(self, v_ce):
        v_max = self.v_max
        K = self.K
        N = self.N

        if v_ce <= 0:
            f_v = (v_max + v_ce) / (v_max - K * v_ce)   

        else:
            f_v = N  + (N-1)*(( - v_max + v_ce)/(7.56*K*v_ce + v_max))

        return f_v
    
    def force_length_relationship_SEE(self, l_see):
        l_slack = self.l_slack
        eps = (l_see - l_slack)/l_slack

        eps_ref = self.eps_ref 

        if eps > 0:
            f_see = (eps/eps_ref)**2
        else:
            f_see = 0

        return f_see
    
    def force_length_relationship_PE(self, l_ce):
        l_opt = self.l_opt

        if l_ce > l_opt:
            f_pe = ((l_ce/l_opt - 1) / self.w)**2
        else:
            f_pe = 0

        return f_pe

    def inverse_force_velocity_CE(self, f_v):
        v_max = self.v_max
        K = self.K
        N = self.N

        if f_v <= 1:
            v_ce = v_max * (f_v - 1) / (1 + K * f_v)    
        elif (f_v > 1)   and (f_v <= N):
            v_ce = (((f_v - 1) * v_max) / (N - 1- 7.56*K*(f_v - N)))
        else:
            v_ce = v_max*(1 + 0.01*(f_v - N))
        return v_ce
    
    def plot_characteristics(self):
        """
        Plots the Force-Length and Force-Velocity curves based on 
        the equations defined in this class.
        """
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Muscle Characteristics: {self.name} (F_max={self.F_max_iso}N)")
        l_scan = np.linspace(0.5 * self.l_opt, 1.8 * self.l_opt, 100)
        
        fl_ce = [self.force_length_relationship_CE(l) * self.F_max_iso for l in l_scan]
        fl_pe = [self.force_length_relationship_PE(l) * self.F_max_iso for l in l_scan]
        fl_total = np.array(fl_ce) + np.array(fl_pe)

        ax[0].plot(l_scan, fl_ce, 'b--', label='Active (CE)')
        ax[0].plot(l_scan, fl_pe, 'g--', label='Passive (PE)')
        ax[0].plot(l_scan, fl_total, 'k-', lw=2, label='Total Isometric')
        ax[0].axvline(self.l_opt, color='r', alpha=0.3, label='L_opt')
        ax[0].set_title("Force-Length Relationship")
        ax[0].set_xlabel("Fiber Length (m)")
        ax[0].set_ylabel("Force (N)")
        ax[0].legend()
        ax[0].grid(True)

        l_see_scan = np.linspace(self.l_slack, 1.08 * self.l_slack, 100)
        fl_see = [self.force_length_relationship_SEE(l) * self.F_max_iso for l in l_see_scan]

        ax[1].plot(l_see_scan, fl_see, 'm-', lw=2)
        ax[1].axvline(self.l_slack, color='r', alpha=0.3, label='L_slack')
        ax[1].set_title("Tendon Stiffness (SEE)")
        ax[1].set_xlabel("Tendon Length (m)")
        ax[1].grid(True)

        v_scan = np.linspace(-self.v_max, 0.5 * self.v_max, 100)
        fv_curve = [self.force_velocity_relationship_CE(v) * self.F_max_iso for v in v_scan]

        ax[2].plot(v_scan, fv_curve, 'r-', lw=2)
        ax[2].axhline(self.F_max_iso, color='k', ls='--', alpha=0.5, label='F_max')
        ax[2].axvline(0, color='k', lw=1)
        ax[2].set_title("Force-Velocity Relationship")
        ax[2].set_xlabel("Velocity (m/s) [-Shortening / +Lengthening]")
        ax[2].set_ylabel("Force (N)")
        ax[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
class MuscleModel:
    def __init__(self, muscle, dt, time, tau):
        self.muscle = muscle
        self.dt = dt
        self.time = time
        self.tau = tau

    def get_act(self, s, time):
        #for constant 's'
        tau = self.tau
        a = s + (1 - s)*(1 - np.exp(-time/tau))
        return a 

    def get_l_m(self, time):
        l_opt = self.muscle.l_opt
        l_slack = self.muscle.l_slack
        l_m = l_opt + l_slack + 0.3*l_opt*np.sin(0.5*2*np.pi*time) 
        return l_m

    def simulate(self, l_m_path, act_path):
        time = self.time
        num_steps = len(time)

        l_ce_0 = l_m_path[0] - self.muscle.l_slack
        l_ce = l_ce_0

        #initialize history lists
        f_m_history = []
        l_ce_history = []   
        v_ce_history = []
        l_see_history = []
    
        for i in range(num_steps):
            l_m = l_m_path[i]
            act = act_path[i]

            #get l_see from current l_ce and l_m
            l_see = l_m - l_ce

            #get forces from lengths
            f_see = self.muscle.force_length_relationship_SEE(l_see)

            f_l_pe = self.muscle.force_length_relationship_PE(l_ce)

            f_l_ce = self.muscle.force_length_relationship_CE(l_ce)

            #since we do not want to use derivatives, get f_ce and and then v_ce and integrate to get l_ce
            f_ce = f_see - f_l_pe
            f_v = f_ce / (act * f_l_ce + 1e-8) #to avoid division by zero

            v_ce = self.muscle.inverse_force_velocity_CE(f_v)
            l_ce = l_ce + v_ce * self.dt #integrate to get new l_ce (explicit Euler)

            #store history
            f_m_history.append(f_ce)
            l_ce_history.append(l_ce)
            v_ce_history.append(v_ce)
            l_see_history.append(l_see)

        return f_m_history, l_ce_history, v_ce_history, l_see_history
    

class MuscleDashboard:
    def __init__(self, muscle, time, l_m, l_ce, v_ce):
        self.muscle = muscle
        self.time = time
        self.l_m = l_m
        self.l_ce = l_ce
        self.v_ce = v_ce

    def animate(self, interval_ms=20):
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        #animated muscle visualization
        ax_phys = fig.add_subplot(gs[0, :])
        ax_phys.set_title("Simulation")
     
        ax_phys.set_xlim(-0.02, np.max(self.l_m) * 1.1)
        ax_phys.set_ylim(-0.1, 0.1)
        #ax_phys.axis('off')
        
        #ce, see, point mass
        ax_phys.axvline(x=0, color='k', linestyle='--', linewidth=2)
        ce_line, = ax_phys.plot([], [], color='firebrick', lw=8, label='CE')
        see_line, = ax_phys.plot([], [], color='royalblue', lw=3, label='SEE')
        mass_point, = ax_phys.plot([], [], 'ko', ms=10)
        time_text = ax_phys.text(0.02, 0.8, '', transform=ax_phys.transAxes)
        ax_phys.plot([self.muscle.l_opt], [0], 'gx', ms=10, mew=2, label='')
        ax_phys.text(self.muscle.l_opt, 0.02, '', ha='center', color='green', fontsize=8)

        rest_len = self.muscle.l_opt + self.muscle.l_slack
        ax_phys.plot([rest_len], [0], 'mx', ms=10, mew=2, label='')
        ax_phys.text(rest_len, 0.02, '', ha='center', color='magenta', fontsize=8)

        # F-L
        ax_fl = fig.add_subplot(gs[1, 0])
        ax_fl.set_title("Force-Length")
        ax_fl.set_xlabel("l_m (m)") 
        ax_fl.set_ylabel("Force")
        
        # F-L
        l_range = np.linspace(0.5 * self.muscle.l_opt, 1.5 * self.muscle.l_opt, 100)
        fl_curve = [self.muscle.force_length_relationship_CE(l) for l in l_range]
        
        ax_fl.plot(l_range, fl_curve, 'k--', alpha=0.5)
        fl_dot, = ax_fl.plot([], [], 'ro', ms=10)

        # F-V
        ax_fv = fig.add_subplot(gs[1, 1])
        ax_fv.set_title("Force-Velocity")
        ax_fv.set_xlabel("v_ce (m/s)") 
        ax_fv.set_ylabel("Force")

        v_range = np.linspace(-self.muscle.v_max, self.muscle.v_max, 100)
        fv_curve = [self.muscle.force_velocity_relationship_CE(v) for v in v_range]
    
        ax_fv.plot(v_range, fv_curve, 'k--', alpha=0.5)
        fv_dot, = ax_fv.plot([], [], 'ro', ms=10)

        def update(frame):
            dt = self.time[1] - self.time[0]
            skip = int(interval_ms / (dt * 1000))
            if skip < 1: skip = 1
            idx = frame * skip
            if idx >= len(self.time): idx = len(self.time) - 1

            curr_l_m = self.l_m[idx]
            curr_l_ce = self.l_ce[idx]
            curr_v_ce = self.v_ce[idx]

            ce_line.set_data([0, curr_l_ce], [0, 0])
            see_line.set_data([curr_l_ce, curr_l_m], [0, 0])
            mass_point.set_data([curr_l_m], [0])
            #time_text.set_text(f'Time: {self.time[idx]:.2f}s')

            fl_val = self.muscle.force_length_relationship_CE(curr_l_ce)
            fl_dot.set_data([curr_l_ce], [fl_val]) 

            fv_val = self.muscle.force_velocity_relationship_CE(curr_v_ce)
            fv_dot.set_data([curr_v_ce], [fv_val])

            return ce_line, see_line, mass_point, fl_dot, fv_dot

        ani = animation.FuncAnimation(fig, update, frames=len(self.time)//20, blit=True, interval=interval_ms)
        plt.tight_layout()
        plt.show()
        return ani


if __name__ == "__main__":

    #define muscle parameters (VAS)
    muscle = Muscle(
        name = "muscle",
        F_max_iso=6000,
        l_opt = 0.08,
        l_slack = 0.23,
        v_max = 12 * 0.08,
        N = 1.5,
        K = 5,
        w = 0.56,
        l_ref=0.5
    )

    dt = 0.001
    duration = 10.0
    time = np.arange(0, duration, dt)
    tau = 0.03
    s = 0.5 
    
    #simulate muscle dynamics
    muscle_model = MuscleModel(muscle, dt, time, tau)
    l_m_path = muscle_model.get_l_m(time)
    act_path = muscle_model.get_act(s, time)  
    f_m, l_ce, v_ce, l_see = muscle_model.simulate(l_m_path, act_path)


    #visualize results
    dashboard = MuscleDashboard(
        muscle=muscle, 
        time=time, 
        l_m=l_m_path, 
        l_ce=l_ce, 
        v_ce=v_ce
    )
    
    dashboard.animate(interval_ms=18)
