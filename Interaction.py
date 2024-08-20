class Interaction(object):
    def __init__(self, args, dataset, mode, env_model=None, env_tokenizer=None):
        self.args = args
        self.dataset = dataset[mode]
        self.max_turn = args.max_turn
        self.conversation = []
        self.cur_conver_step = 0
        self.test_num = 0
        self.mode = mode
        set_random_seed(args.seed)

    def persuader_step(self, persuasive_strategy):
        messages = message_format[self.args.data_name](self.case, 'persuader', self.conversation, persuasive_strategy)
        persuader_response = self.generate_response(self.args.persuader, messages, persuader_role[self.args.data_name])
        persuader_response = self.postprocess_response(persuader_response, resister_role[self.args.data_name])
        self.conversation.append({"role":persuader_role[self.args.data_name],"content":response})
        print(self.conversation[-1])
        return self.conversation

    def resister_step(self, resistant_strategy):
        messages = message_format[self.args.data_name](self.case, 'resister', self.conversation,resistant_strategy)
        resister_response = self.generate_response(self.args.resister, messages, resister_role[self.args.data_name])
        resister_response = self.postprocess_response(resister_response, persuader_role[self.args.data_name])
        self.conversation.append({"role":resister_role[self.args.data_name], "content":resister_response})
        print(self.conversation[-1])
        return self.conversation
    

    def generate_response(self, model, messages, role):
        messages = chatgpt_prompt(messages, role)
        output = query_openai_model(
            api_key=YOUR_API_KEY,
            messages=messages,
            model=model,
            max_tokens=self.args.max_new_tokens,
            temperature=temperature
        )
        return output
