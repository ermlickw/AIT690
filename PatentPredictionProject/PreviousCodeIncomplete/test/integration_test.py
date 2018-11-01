from factory import Factory
from config import Config


TITLE = 'SYSTEM AND METHOD FOR ESTIMATING THE POSITION AND ORIENTATION OF A MOBILE COMMUNICATIONS DEVICE IN A BEACON-BASED POSITIONING SYSTEM'
ABSTRACT = 'An example of a lighting device including a light source, a modulator and a processor. The processor is configured to control the light source to emit light for general illumination and control the modulator to modulate the intensity of the emitted light to superimpose at least two sinusoids. Frequencies of the at least two sinusoids enable a mobile device to infer the physical location of the lighting device.'
CLAIMS = '1. A lighting device, comprising: a light source; a modulator coupled to the light source; and a processor coupled to the modulator and configured to: control the light source to emit visible light for general illumination within a space; and control the modulator to: modulate the intensity of visible light emitted by the light source based on a signal comprising at least two superimposed sinusoids and in accordance with at least two frequencies of the at least two superimposed sinusoids such that the at least two superimposed sinusoids are simultaneously broadcast; vary the frequency of a first of the at least two superimposed sinusoids, between a number of varied frequencies and within a modulation range, during each of a plurality of cycles, each cycle corresponding to a timeframe; maintain each respective varied frequency of the first of the at least two superimposed sinusoids during a respective cycle for some period of time, each period of time being a fraction of the respective timeframe corresponding to the respective cycle such that the collection of time periods for the respective number of varied frequencies of the respective cycle equals the respective timeframe; and repeat the plurality of cycles some number of times.'


class IntegrationTest(object):

    @staticmethod
    def test_predict():
        config_info = Config()
        f = Factory(config_info)
        f.classify.load_classifier('Perceptron2016-06-11')
        feature_vector = f.evaluate(TITLE, ABSTRACT, CLAIMS)
        f.predict(feature_vector)

if __name__ == '__main__':
    IntegrationTest.test_predict()
