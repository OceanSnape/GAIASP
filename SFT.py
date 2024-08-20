def train(args, train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, len(args.device_id))
    train_dataloader = DataLoader(DataFrame(train_dataset, args), batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if len(args.device_id) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_id)
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    utils.set_random_seed(args.seed)  
    global_step = 0
    best_f1 = 0 
    for e in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            inputs = {'input_ids': batch['input_ids'].to(args.device),
                        'attention_mask': batch['attention_mask'].to(args.device),
                        'labels': batch['labels'].to(args.device)}
            outputs = model(**inputs)
            loss = outputs#[0]  
            if len(args.device_id) > 1:
                loss = loss.mean()  
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()
                optimizer.step()
                model.zero_grad()
            global_step += 1
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / (step+1), global_step)
        print('loss: {}'.format((tr_loss - logging_loss) / (step+1)))
        logging_loss = tr_loss
        results = evaluate(args, model, tokenizer, save_output=True)               
        if results[0] > best_f1:
            best_f1 = results[0]
            output_dir = os.path.join(args.output_dir, 'best_checkpoint')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                            'module') else model 
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    tb_writer.close()
    return global_step, tr_loss / global_step


