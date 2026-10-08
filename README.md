# qwen3-mini


      python3 -m venv venv
      source venv/bin/activate
      pip install --upgrade pip
      pip install torch numpy datasets tiktoken transformers 



      nohup python -u model.py > model.log 2>&1 &

     nohup python -u model.py --resume > model.log 2>&1 &
