import random




class WeatherSimulator():
    def __init__(self):
        self.transition_probs = {
            'sunny': {'sunny': 0.8, 'rainy': 0.2},
            'rainy': {'sunny': 0.4, 'rainy': 0.6}
        }

        self.emission_probs = {
            'sunny': {'happy': 0.8, 'sad': 0.2},
            'rainy': {'happy': 0.4, 'sad': 0.6}
        }

        self.prior_probs = {
            'sunny': 0.67,
            'rainy': 0.33
        }



class WeatherHMM():
    def __init__(self, prior_probs, transition_probs, emission_probs):
        self.prior_probs = prior_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs


    def viterbi(self, observations):
        daily_estimations = {}
        i = 1
        for observation in observations:
            if i == 1:
                daily_estimations[i] = {
                    'sunny': float(self.prior_probs['sunny'] * self.emission_probs['sunny'][observation]),
                    'rainy': float(self.prior_probs['rainy'] * self.emission_probs['rainy'][observation])
                }
            else:
                    
                daily_estimations[i] = {
                    'sunny': max(float(daily_estimations[i-1]['sunny'] * self.emission_probs['sunny'][observation] * self.transition_probs['sunny']['sunny']),
                                float(daily_estimations[i-1]['rainy'] * self.emission_probs['sunny'][observation] * self.transition_probs['rainy']['sunny'])),
                    
                    'rainy': max(float(daily_estimations[i-1]['sunny'] * self.emission_probs['rainy'][observation] * self.transition_probs['sunny']['rainy']),
                                float(daily_estimations[i-1]['rainy'] * self.emission_probs['rainy'][observation] * self.transition_probs['rainy']['rainy']))
                }
            i += 1
        return daily_estimations
    

    def max_likelihood_estimate(self, probabilities):
        predictions = {}
        for k in probabilities:
            print(f"Day {k} | Sunny: {probabilities[k]['sunny']} | Rainy: {probabilities[k]['rainy']}")
            predictions[k] = 'sunny' if probabilities[k]['sunny'] > probabilities[k]['rainy'] else 'rainy'


        return predictions
            

    



# Hidden states can be either Sunny or Rainy

observations = ['happy', 'happy', 'sad', 'sad', 'sad', 'happy']

weather_sim = WeatherSimulator()
hmm = WeatherHMM(weather_sim.prior_probs, weather_sim.transition_probs, weather_sim.emission_probs)

daily_estimations = hmm.viterbi(observations)
predictions = hmm.max_likelihood_estimate(daily_estimations)
print(predictions)




