from google.cloud import texttospeech

import os



# Authenticate with your API key (make sure to set GOOGLE_APPLICATION_CREDENTIALS environment variable)

#export
GOOGLE_APPLICATION_CREDENTIALS="/home/lab_phe3223/AI_voice/credentials/verdant-oven-445617-r4-850725ab4150.json"




client = texttospeech.TextToSpeechClient()

# The text to synthesize

text = '''When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is , according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow." Throughout the centuries people have explained the rainbow in various ways. Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain. The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain. Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed. The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows. If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue. '''
# List of voices you can use

voices = [

    'en-US-Wavenet-A', 'en-US-Wavenet-B', 'en-US-Wavenet-C', 'en-US-Wavenet-D',

    'en-US-Wavenet-E', 'en-US-Wavenet-F', 'en-GB-Wavenet-A', 'en-GB-Wavenet-B'

]



# Output directory

output_dir = "normal_synthesized_audio_2000"

if not os.path.exists(output_dir):

    os.makedirs(output_dir)



for i in range(2000):

    # Choose a random voice from the list

    voice_name = voices[i % len(voices)]

    voice = texttospeech.VoiceSelectionParams(

        language_code='en-US',

        name=voice_name

    )

 # Audio configuration

    audio_config = texttospeech.AudioConfig(

        audio_encoding=texttospeech.AudioEncoding.LINEAR16

    )
# Request to synthesize speech

    synthesis_input = texttospeech.SynthesisInput(text=text)



    # Generate the audio

    response = client.synthesize_speech(

        input=synthesis_input, voice=voice, audio_config=audio_config

    )



    # Write the audio to a file

    output_file = f"{output_dir}/voice_{i+1}.wav"

    with open(output_file, 'wb') as out:

        out.write(response.audio_content)



    print(f"Generated: {output_file}")
