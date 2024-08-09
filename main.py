import sys, getopt
from stable_whisper import load_model
from stable_whisper import results_to_sentence_vtt
from stable_whisper import stabilize_timestamps

def main(argv):
  inputfile = ''
  outputfile = ''
  model_type = 'base'
  try:
    opts, args = getopt.getopt(argv,"hi:o:m:",["ifile=","ofile=","model="])
  except getopt.GetoptError:
    print('test.py -i <inputfile> -o <outputfile>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
        print('main.py -i <inputfile> -o <outputfile> -m <model_type>')
        print('\tmodel_type: tiny, base, small, medium, large  (base is default)')
        sys.exit()
    elif opt in ("-i", "--ifile"):
        inputfile = arg
    elif opt in ("-o", "--ofile"):
        outputfile = arg
    elif opt in ("-m", "--model"):
        model_type = arg
  print('Input file is: ', inputfile)
  print('Output file is: ', outputfile)
  model = load_model(model_type)
  # modified model should run just like the regular model but with additional hyperparameters and extra data in results
  results = model.transcribe(inputfile)

  options = dict(language="french", beam_size=5, best_of=5)
  translate_options = dict(task="translate", **options)
  resFrench = model.transcribe(inputfile, translate_options)
  results_to_sentence_vtt(resFrench, "out_french.vtt")

  # after you get results from modified model
  results_to_sentence_vtt(results, outputfile)
 
  
# Using the special variable 
# __name__
if __name__=="__main__":
  main(sys.argv[1:])
