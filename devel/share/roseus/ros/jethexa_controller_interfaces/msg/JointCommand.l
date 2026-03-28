;; Auto-generated. Do not edit!


(when (boundp 'jethexa_controller_interfaces::JointCommand)
  (if (not (find-package "JETHEXA_CONTROLLER_INTERFACES"))
    (make-package "JETHEXA_CONTROLLER_INTERFACES"))
  (shadow 'JointCommand (find-package "JETHEXA_CONTROLLER_INTERFACES")))
(unless (find-package "JETHEXA_CONTROLLER_INTERFACES::JOINTCOMMAND")
  (make-package "JETHEXA_CONTROLLER_INTERFACES::JOINTCOMMAND"))

(in-package "ROS")
;;//! \htmlinclude JointCommand.msg.html


(defclass jethexa_controller_interfaces::JointCommand
  :super ros::object
  :slots (_target _duration ))

(defmethod jethexa_controller_interfaces::JointCommand
  (:init
   (&key
    ((:target __target) (make-array 18 :initial-element 0.0 :element-type :float))
    ((:duration __duration) 0.0)
    )
   (send-super :init)
   (setq _target __target)
   (setq _duration (float __duration))
   self)
  (:target
   (&optional __target)
   (if __target (setq _target __target)) _target)
  (:duration
   (&optional __duration)
   (if __duration (setq _duration __duration)) _duration)
  (:serialization-length
   ()
   (+
    ;; float32[18] _target
    (* 4    18)
    ;; float32 _duration
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[18] _target
     (dotimes (i 18)
       (sys::poke (elt _target i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;; float32 _duration
       (sys::poke _duration (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[18] _target
   (dotimes (i (length _target))
     (setf (elt _target i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     )
   ;; float32 _duration
     (setq _duration (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get jethexa_controller_interfaces::JointCommand :md5sum-) "10ef9426383df380694d1e1ce8eec8d2")
(setf (get jethexa_controller_interfaces::JointCommand :datatype-) "jethexa_controller_interfaces/JointCommand")
(setf (get jethexa_controller_interfaces::JointCommand :definition-)
      "float32[18] target
float32 duration
")



(provide :jethexa_controller_interfaces/JointCommand "10ef9426383df380694d1e1ce8eec8d2")


