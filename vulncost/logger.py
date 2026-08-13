

# Structure of the classes
# Run 
#   Sample
#     Repeat
#       Call Number  

class CallRecord:

    def __init__(self, model, 
                 approach, context_config, 
                 agent, sample_id, repeat, call_number,
                 prompt, response, 
                 input_tokens, output_tokens, prediction, 
                 label, status, duration_s, temperature, seed):
        self.model = model
        self.approach = approach # if it is context or llm vs agent comparison
        self.context_config = context_config # configuration of context
        self.agent = agent # agent infrastructure
        self.sample_id = sample_id # idx from PrimeVul
        self.repeat = repeat 
        self.call_number = call_number
        self.prompt = prompt 
        self.response = response
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.prediction = prediction
        self.label = label
        self.status = status
        self.duration_s = duration_s
        self.temperature = temperature
        self.seed = seed