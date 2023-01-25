import numpy as np
import scipy.stats
from ipywidgets import Button, HBox


class MultiArmedBandit():
    def __init__(self, K=5, rng=None):
        self.rng = np.random if rng is None else rng
        self.K = K
         
        # Generate means in [0,10] uniformly
        self.mus = [10*self.rng.rand() for _ in range(K)]
        # Observation noise is clipped Gaussian
        self.sigma = 1
        self.noise = lambda: np.clip(self.rng.randn(),-10,10)
    
    def pull(self, arm):
        return self.mus[arm] + self.noise()
    
class RewardHistory():

    def __init__(self, actions):
        self.history = np.empty([0,2])
        self.actions = actions
        self.per_action_history = {action: [] for action in actions}
        self.ci = {action: None for action in actions} # mean, upper, lower
        self.T = 0
    
    def record(self, action, reward):
        self.history = np.append(self.history, [[action, reward]], axis=0)
        self.per_action_history[action].append(reward)
        self.T += 1

    def compute_ci(self, mean_only=False):
        for action in self.actions:
            N = len(self.per_action_history[action])
            mean = np.mean(self.per_action_history[action]) if N > 0 else None
            if not mean_only and N > 0:
                lower, upper = scipy.stats.norm.interval(0.95, loc=mean, 
                                                         scale=1/np.sqrt(N))
            else:
                lower, upper = None, None
            self.ci[action] = (mean, upper, lower)
            
    def get_unexplored_actions(self, N=0):
        return [a for a in self.actions if len(self.per_action_history[a])<=N]
            
    def get_smallest_N(self):
        min_N = min([len(self.per_action_history[a]) for a in self.actions])
        return [a for a in self.actions if len(self.per_action_history[a]) <= min_N][0]
         
    def get_means(self):
        self.compute_ci()
        return np.array([self.ci[a][0] for a in self.actions])

    def get_highest_mean(self):
        self.compute_ci()
        max_r = max([self.ci[a][0] for a in self.actions])
        return [a for a in self.actions if self.ci[a][0] >= max_r][0]
    
    def get_ucb(self):
        self.compute_ci()
        ucb = max([self.ci[a][1] for a in self.actions])
        return [a for a in self.actions if self.ci[a][0] >= ucb][0]
    
    def get_widest_ci(self):
        return self.get_smallest_N()

def update_plot(axs, hist, K, ci=False, baseline=None):
    ax = axs[0]
    ax.clear()
    ax.scatter(hist.history[:,0], hist.history[:,1], alpha=0.75)
    ax.set_xlim([-1,K])
    ax.set_ylim([min(0, np.amin(hist.history[:,1])),max(12, np.amax(hist.history[:,1]))])
    ax.set_title("observed rewards per arm")
    ax = axs[1]
    ax.clear()
    if baseline is None:
        y = np.cumsum(hist.history[:,1])
        title = "cumulative reward"
    else:
        y = baseline-np.cumsum(hist.history[:,1])
        title = "regret"
    ax.plot(np.arange(0,hist.T+1), np.append([0],y))
    ax.set_xlim([0,hist.T])
    ax.set_ylim([0,np.amax(y)])
    ax.set_title(title)
    if ci:
        ax = axs[0]
        xs = [x for x in hist.actions if hist.ci[x][0] is not None]
        ys = [hist.ci[x][0] for x in xs]
        y_errs = [hist.ci[x][1]-hist.ci[x][2] if hist.ci[x][1] is not None else 0 for x in xs]
        ax.errorbar(x=xs, y=ys, yerr=y_errs, color="black", capsize=3,
             linestyle="None",marker="s", markersize=7, mfc="black", mec="black", alpha=0.25)

class InteractivePlot():
    def __init__(self, mab, hist, axs, mean=False, ci=False):
        self.mab = mab
        self.hist = hist
        arm_buttons = [Button(description=str(arm)) for arm in np.arange(mab.K)]
        reveal_button = Button(description='Reveal')
        self.combined = HBox([items for items in arm_buttons] + [reveal_button])
        
        self.axs = axs
        
        self.mean = mean
        self.ci = ci
        
        for n in range(mab.K):
            arm_buttons[n].on_click(self.upon_clicked)
        reveal_button.on_click(self.upon_reveal)
        
    def upon_clicked(self, btn):
        arm = int(btn.description)
        reward = self.mab.pull(arm)
        self.hist.record(arm, reward)
        if self.mean and not self.ci:
            self.hist.compute_ci(mean_only=True)
        elif self.ci:
            self.hist.compute_ci()
        update_plot(self.axs, self.hist, self.mab.K,ci=(self.mean or self.ci)) 
    
    def upon_reveal(self, b):
        self.axs[0].scatter(np.arange(self.mab.K), self.mab.mus, marker="*")
        self.axs[1].plot(np.arange(self.hist.T), np.amax(self.mab.mus)*np.arange(self.hist.T))

def greedy_policy(hist):
    unexplored = hist.get_unexplored_actions()
    if len(unexplored) > 0:
        return np.random.choice(unexplored)
    return hist.get_highest_mean()