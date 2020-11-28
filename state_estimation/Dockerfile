FROM duckietown/challenge-aido_lf-baseline-duckietown:daffy-amd64

RUN pip3 install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN  pip3 install --use-feature=2020-resolver -r .requirements.txt

RUN mkdir /code/exercise_ws

RUN useradd -ms /bin/bash admin

COPY exercise_ws /code/exercise_ws
USER root
RUN chmod -R +x /code/exercise_ws
USER admin
COPY launchers /code

CMD ["bash", "/code/submit.sh"]
