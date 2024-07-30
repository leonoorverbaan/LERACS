# LERACS
**LERACS: LLM-Enhanced Robotic Affordance and Control System**

The system that is designed, abbreviated for simplicity LERACS, provides a vision and control method for manipulation affordance with the use of OpenAI's ChatGPT API in a few-shot setting. In the vision method images are translated into vision input information for the control method, and perceptually grounded. The control method uses this vision input information together with natural-language instructions, while mitigating the impact of the token limit from ChatGPT, to translate this into executable robot actions in a planned task order. This system uses customizable input prompts for both environment and task decomposition generation using ChatGPT. This system therefore generates **multi-step task plans**, an **execution query** from the task plan for the robot arm to perform, an **updated environment** from instructions and image environment data, and a **visualized scene with affordable objects for manipulation**. The figure underneath displays these decomposed parts of the system. 

![System Flowchart](Images/system_flowchart.png)


## Required Software and Hardware
### Required hardware:
- Franka Emika Research 3 Robot arm
- External PC (Ubuntu 20.04)
- Objects with ArUco markers

### Franka Emika Research 3 Robot
1. Mount the robot securely to a flat surface.
2. Connect the power cable and the controller to the robot.
3. Power on the robot, connect the Ethernet cable to the external PC and go to the interface of the IP address.
4. Unlock all the joints and activate the FCI.

### External PC (Ubuntu 20.04)
As the robot arm does not have an internal computer capable of running the ROS components, a separate machine is required. This machine can be anything capable of running Ubuntu 20.04, such as an Intel NUC or a laptop.

Make sure that Ubuntu is installed with a realtime kernel. This gives the opportunity to control the robot arm in real-time. Installation info can be found [here](https://frankaemika.github.io/docs/installation_linux.html).

### Ethernet Connection
Connect the Ethernet cable to the port on the robot and to a port on the PC. So that the PC can connect to the robot driver via the Ethernet connection, the wired connection has to be configured in the following way:

### ROS
This project was developed using ROS Noetic. It may work with other ROS releases, but it is recommended to set up a Noetic environment on the external PC, using these [instructions](http://wiki.ros.org/noetic/Installation/Ubuntu).



## Installation

1. Create a new workspace directory that includes a `src` directory.
2. Clone the following repositories into the same `catkin_ws`:

    ```sh
    mkdir ~/catkin_ws/src
    cd ~/catkin_ws/src
    git clone git@github.com:Alliander/LERACS.git
    cd LERACS.git
    sh install.sh ~/path/to/catkin_ws
    ```

3. Test the setup by running `roslaunch mrirac fr3_sim.launch`. This launches the Gazebo simulation.

If the setup was successful, you should be able to set a pose goal using the rviz interface and the simulated robot will move to that position once `plan` and `execute` is pressed.


## Credits

This project was developed as part of the Master Thesis for my (Leonoor Verbaan) MSc. Robotics at the TU Delft. The project was supervised by Yke Bauke Eisema (Cognitive Robotics, TU Delft) and Remco van Leeuwen (Research Center for Digital Technologies, Alliander).

This package is an adaption of the original [ChatGPT-Robot-Manipulation-Prompts](https://github.com/microsoft/ChatGPT-Robot-Manipulation-Prompts). Created by Matthew Hanlon and supervised by Eric Vollenweider from the ETH Zurich.


## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit [https://cla.opensource.microsoft.com](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general.aspx). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.


