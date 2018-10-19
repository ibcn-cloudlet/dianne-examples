package dianne.examples.onnx;

import java.util.Map;

import org.osgi.service.component.annotations.Component;
import org.osgi.service.component.annotations.Reference;

import be.iminds.iot.dianne.api.nn.NeuralNetwork;

@Component(immediate=true)
public class DianneServiceExample {
	
	@Reference
	void setNeuralNetwork(NeuralNetwork nn, Map<String, Object> properties) {
		System.out.println("Neural Network service found!");
		properties.entrySet()
			.stream().filter(e -> e.getKey().startsWith("nn."))
			.forEach(e -> System.out.println("* "+e.getKey()+" : "+e.getValue()));
	}
	
	
}
