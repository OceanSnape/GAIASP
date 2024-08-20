class Plnner(nn.Module):
    def __init__(self, args, config, tokenizer):
        super().__init__()
        self.policy = model[args.model_name].from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, cache_dir=args.cache_dir)
        self.dropout = nn.Dropout(0.5)
        self.act = sorted(list(act[args.data_name].keys()))
        self.classifier = nn.Linear(config.hidden_size, len(self.act))
        self.tokenizer = tokenizer
        self.optimizer = AdamW(
            self.parameters(), lr=args.learning_rate
        )
        self.eps = np.finfo(np.float32).eps.item()
        self.config = config
        self.args = args
        self.saved_log_probs = []
        self.rewards = []


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, len(self.act)), labels.view(-1))
            return loss
        else:
            return F.softmax(logits, dim=-1)

    def select_strategy(self, state, is_test=False):
        inp = self.build_input(state)
        inp = torch.tensor(inp).long()

        outputs = self.policy(inp)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = nn.functional.softmax(logits, dim=1)
        m = Categorical(probs)
        if is_test:
            strategy = probs.argmax().item()
        else:
            strategy = m.sample()
            self.saved_log_probs.append(m.log_prob(strategy))
        return self.act[strategy]

    def rl_optimize(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + self.args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        if rewards.shape[0] > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
        return policy_loss.data
