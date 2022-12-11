ARG WANDB_KEY
FROM python:3.10 as base
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install Gymnasium gymnasium[box2d]
RUN pip3 install moviepy numpy hydra-core  wandb
COPY . .
RUN echo $WANDB_KEY
RUN echo test
RUN wandb login $WANDB_KEY
ENTRYPOINT [ "python", "baseline.py" ]