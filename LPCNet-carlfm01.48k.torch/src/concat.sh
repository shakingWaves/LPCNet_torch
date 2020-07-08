# Place in 16k-LP7 from TSPSpeech.iso and run to concatenate wave files
# into one headerless training file
for dir in Wave48_9900
do
echo ${dir}
for i in /data/lixiaobo5/Synthesis/BZNSYP/${dir}/*.wav
do
sox $i -r 48000 -c 1 -t sw -
done >> /data/lixiaobo5/Synthesis/BZNSYP/input.48k.s16
done
