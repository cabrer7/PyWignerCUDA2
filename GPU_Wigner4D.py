#...................................................................
import ctypes
import numpy as np
import scipy.fftpack as fftpack
import h5py
import time
import sys
from scipy.special import laguerre
from scipy.special import genlaguerre
from scipy.special import legendre

#from pyfft.cuda import Plan
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.driver as cuda
import pycuda.autoinit

import pycuda.reduction as reduction
from pycuda.elementwise import ElementwiseKernel


from pycuda.compiler import SourceModule


#.................................................................
#            FAFT library
#.................................................................

DIR_BASE_FAFT = "/home/rcabrera/Documents/source/Python/PyWignerCUDA2/FAFT/"

# FAFT 64-points
_faft128_4D = ctypes.cdll.LoadLibrary( DIR_BASE_FAFT+'FAFT128_4D_Z2Z.so' )
_faft128_4D.FAFT128_4D_Z2Z.restype = int
_faft128_4D.FAFT128_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double ]

cuda_faft64 = _faft128_4D.FAFT128_4D_Z2Z


# FAFT 128-points
_faft256_4D = ctypes.cdll.LoadLibrary( DIR_BASE_FAFT+'FAFT256_4D_Z2Z.so' )
_faft256_4D.FAFT256_4D_Z2Z.restype = int
_faft256_4D.FAFT256_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double ]

cuda_faft128 = _faft256_4D.FAFT256_4D_Z2Z

#..................................................................


class GPU_Wigner4D:
	"""
	
	"""
	def __init__( self ):


		self.gridDIM_p_y = 64      #axis 0
		self.gridDIM_y   = 64      #axis 1    
		self.gridDIM_p_x = 128     #axis 2
		self.gridDIM_x   = 128     #axis 3

		size = self.gridDIM_y*self.gridDIM_p_y*self.gridDIM_x*self.gridDIM_p_x

		self.FAFT_axes0 = 0
		self.FAFT_axes1 = 1
		self.FAFT_axes2 = 2
		self.FAFT_axes3 = 3

		#m = size

		self.FAFT_segment_axes0 = 0
		self.FAFT_segment_axes1 = 0
		self.FAFT_segment_axes2 = 0
		self.FAFT_segment_axes3 = 0

		self.NF = 1

		# Phase space step size 
		self.dp_y  = 2*self.p_y_amplitude/float(self.gridDIM_p_y)           #axis 0
		self.dy    = 2*self.y_amplitude  /float(self.gridDIM_y)             #axis 1
		self.dp_x  = 2*self.p_x_amplitude/float(self.gridDIM_p_x)           #axis 2
		self.dx    = 2*self.x_amplitude  /float(self.gridDIM_x)             #axis 3  

		# Ambiguity space step size
		self.dtheta_y  = 2*self.theta_y_amplitude /float(self.gridDIM_y)     #axis 0
		self.dlambda_y = 2*self.lambda_y_amplitude/float(self.gridDIM_p_y)   #axis 1
		self.dtheta_x  = 2*self.theta_x_amplitude /float(self.gridDIM_x)     #axis 2
		self.dlambda_x = 2*self.lambda_x_amplitude/float(self.gridDIM_p_x)   #axis 3

		# delta parameters
		self.delta_p_y =   self.dp_y*self.dtheta_y /(2*np.pi)     #axis 0
		self.delta_y  =    self.dy  *self.dlambda_y/(2*np.pi)     #axis 1
		self.delta_p_x =   self.dp_x*self.dtheta_x /(2*np.pi)     #axis 2
		self.delta_x  =    self.dx  *self.dlambda_x/(2*np.pi)     #axis 3

		# Phase space
		self.p_y_range = np.linspace( -self.p_y_amplitude, self.p_y_amplitude-self.dp_y,  self.gridDIM_p_y)    #axis 0
		self.y_range   = np.linspace( -self.y_amplitude,   self.y_amplitude  -self.dy,    self.gridDIM_y  )    #axis 1
		self.p_x_range = np.linspace( -self.p_x_amplitude, self.p_x_amplitude-self.dp_x,  self.gridDIM_p_x)    #axis 2
		self.x_range   = np.linspace( -self.x_amplitude,   self.x_amplitude  -self.dx,    self.gridDIM_x  )    #axis 3

		# Ambiguity space range
		self.theta_y_range = np.linspace(-self.theta_y_amplitude,self.theta_y_amplitude-self.dtheta_y,self.gridDIM_y  )    #0
		self.lambda_y_range= np.linspace(-self.lambda_y_amplitude,self.lambda_y_amplitude-self.dlambda_y, self.gridDIM_p_y)#1
		self.theta_x_range = np.linspace(-self.theta_x_amplitude,self.theta_x_amplitude -self.dtheta_x,  self.gridDIM_x  ) #2
		self.lambda_x_range= np.linspace(-self.lambda_x_amplitude,self.lambda_x_amplitude-self.dlambda_x,self.gridDIM_p_x) #3

		# Grid 
		self.y   =   self.y_range[ np.newaxis, :, np.newaxis, np.newaxis ]   #axis 1
		self.p_y = self.p_y_range[ :, np.newaxis, np.newaxis, np.newaxis ]   #axis 0
		self.p_x = self.p_x_range[ np.newaxis, np.newaxis, :, np.newaxis ]   #axis 2
		self.x   =   self.x_range[ np.newaxis, np.newaxis, np.newaxis, : ]   #axis 3
		
		self.CUDA_constants =  '\n'

		self.CUDA_constants += '__device__ double dt   = %f;   '%self.dt
		self.CUDA_constants += '__device__ double mass = %f; \n'%self.mass

		self.CUDA_constants += '__device__ double dp_y   = %f; '%self.dp_y
		self.CUDA_constants += '__device__ double dy     = %f; '%self.dy
		self.CUDA_constants += '__device__ double dp_x   = %f; '%self.dp_x
		self.CUDA_constants += '__device__ double dx     = %f; \n'%self.dx

		self.CUDA_constants += '__device__ double dtheta_y   = %f; '%self.dtheta_y
		self.CUDA_constants += '__device__ double dlambda_y  = %f; '%self.dlambda_y
		self.CUDA_constants += '__device__ double dtheta_x   = %f; '%self.dtheta_x
		self.CUDA_constants += '__device__ double dlambda_x  = %f; \n'%self.dlambda_x

		self.CUDA_constants += '__device__ int gridDIM_x = %d; '%self.gridDIM_x
		self.CUDA_constants += '__device__ int gridDIM_y = %d; '%self.gridDIM_y

		try :
			self.CUDA_constants += '__device__ double D_lambda_x = %f; '%self.D_lambda_x
			self.CUDA_constants += '__device__ double D_lambda_y = %f; '%self.D_lambda_y
			self.CUDA_constants += '__device__ double D_theta_x  = %f; '%self.D_theta_x
			self.CUDA_constants += '__device__ double D_theta_y  = %f; \n'%self.D_theta_y
		except AttributeError:
			pass

		self.CUDA_constants +=  '\n'		

		print self.CUDA_constants

		#...........................................................................

		print '         GPU memory Total               ', pycuda.driver.mem_get_info()[1]/float(2**30) , 'GB'
		print '         GPU memory Free  (Before)      ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'

		self.W_init_gpu = gpuarray.zeros(
				( self.gridDIM_p_y, self.gridDIM_y, self.gridDIM_p_x, self.gridDIM_x ), dtype=np.complex128 )
		print '         GPU memory Free  (After)       ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'	

		#............................................................................

		self.indexUnpack_x_p_string = """
			int i_x   = i%gridDIM_x;
			int i_p_x = (i/gridDIM_x) % gridDIM_x;
			int i_y   = (i/(gridDIM_x*gridDIM_x)) % gridDIM_y;
			int i_p_y = i/(gridDIM_x*gridDIM_x*gridDIM_y);

			double x   = dx  *( i_x   - gridDIM_x/2 );
			double p_x = dp_x*( i_p_x - gridDIM_x/2 );
			double y   = dy  *( i_y   - gridDIM_y/2 );
			double p_y = dp_y*( i_p_y - gridDIM_y/2 );			
			"""

		self.indexUnpack_lambda_theta_string = """
			int i_x   = i%gridDIM_x;
			int i_p_x = (i/gridDIM_x) % gridDIM_x;
			int i_y   = (i/(gridDIM_x*gridDIM_x)) % gridDIM_y;
			int i_p_y = i/(gridDIM_x*gridDIM_x*gridDIM_y);

			double lambda_x  = dlambda_x * ( i_x   - gridDIM_x/2 );
			double theta_x   = dtheta_x  * ( i_p_x - gridDIM_x/2 );
			double lambda_y  = dlambda_y * ( i_y   - gridDIM_y/2 );
			double theta_y   = dtheta_y  * ( i_p_y - gridDIM_y/2 );			
			"""

		self.indexUnpack_lambda_p_string = """
			int i_x    = i%gridDIM_x;
			int i_p_x  = (i/gridDIM_x) % gridDIM_x;
			int i_y    = (i/(gridDIM_x*gridDIM_x)) % gridDIM_y;
			int i_p_y  = i/(gridDIM_x*gridDIM_x*gridDIM_y);

			double lambda_x   = dx  *( i_x   - gridDIM_x/2 );
			double p_x        = dp_x*( i_p_x - gridDIM_x/2 );
			double lambda_y   = dy  *( i_y   - gridDIM_y/2 );
			double p_y        = dp_y*( i_p_y - gridDIM_y/2 );			
			"""
		self.indexUnpack_x_theta_string = """
			int i_x   = i%gridDIM_x;
			int i_p_x = (i/gridDIM_x) % gridDIM_x;
			int i_y   = (i/(gridDIM_x*gridDIM_x)) % gridDIM_y;
			int i_p_y = i/(gridDIM_x*gridDIM_x*gridDIM_y);

			double x       = dx      *( i_x   - gridDIM_x/2 );
			double theta_x = dtheta_x*( i_p_x - gridDIM_x/2 );
			double y       = dy      *( i_y   - gridDIM_y/2 );
			double theta_y = dtheta_y*( i_p_y - gridDIM_y/2 );	
			"""

		#...............................................................................................		

		self.Gaussian_GPU = ElementwiseKernel(
		    """pycuda::complex<double> *W , 
				double    mu_p_y, double    mu_y, double    mu_p_x, double    mu_x , 
				double sigma_p_y, double sigma_y, double sigma_p_x, double sigma_x """
		    ,
		    self.indexUnpack_x_p_string + """
			double temp =   exp(-0.5*( x   - mu_x   )*( x   - mu_x   )/( sigma_x   * sigma_x   )  );
			       temp *=  exp(-0.5*( y   - mu_y   )*( y   - mu_y   )/( sigma_y   * sigma_y   )  );
			       temp *=	exp(-0.5*( p_x - mu_p_x )*( p_x - mu_p_x )/( sigma_p_x * sigma_p_x )  );
			       temp *=	exp(-0.5*( p_y - mu_p_y )*( p_y - mu_p_y )/( sigma_p_y * sigma_p_y )  );

			W[i] = pycuda::complex<double>(  temp , 0. ); 

					"""
		    ,"Gaussian",  preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants )	

		#
		self.HOscillatorGound_GPU = ElementwiseKernel(
		    """pycuda::complex<double> *W, 
			double   x_mu, double    y_mu,
                        double p_x_mu, double  p_y_mu,
			double omega_x, double omega_y, double mass"""
		   ,
		   self.indexUnpack_x_p_string + """
			double temp  = (mass*pow( omega_x*(x-x_mu) ,2) + pow(p_x-p_x_mu,2)/mass)/omega_x;   
			       temp += (mass*pow( omega_y*(y-y_mu) ,2) + pow(p_y-p_y_mu,2)/mass)/omega_y;

			W[i] = pycuda::complex<double>(  exp(-temp) , 0. ); 
					          """
		   ,"Gaussian",  preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants )	

		# Kinetic propagator ................................................................................

		kineticStringC = '__device__ double K(double p_x, double p_y){ \n return '+self.kineticString+';\n}'

		self.exp_p_lambda_GPU = ElementwiseKernel(
			""" pycuda::complex<double> *B """
			,
			self.indexUnpack_lambda_p_string + """ 				
			double  r  = exp( - dt*D_lambda_y * lambda_x*lambda_x );
	 			r *= exp( - dt*D_lambda_y * lambda_y*lambda_y );
 
			double phase  = dt*K(p_x + 0.5*lambda_x, p_y + 0.5*lambda_y) - dt*K(p_x - 0.5*lambda_x, p_y - 0.5*lambda_y);
			B[i] *= pycuda::complex<double>( r*cos(phase), -r*sin(phase) );

			"""
  		       ,"exp_p_lambda_GPU", preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants +kineticStringC )  
		
		#  Potential propagator ..............................................................................

		potentialStringC = '__device__ double V(double x, double y){ \n return '+self.potentialString+';\n}'
				

		self.exp_x_theta_GPU = ElementwiseKernel(
			""" pycuda::complex<double> *B """
			,
			self.indexUnpack_x_theta_string + """ 
			double phase  = dt*V(x-0.5*theta_x , y-0.5*theta_y) - dt*V( x+0.5*theta_x , y+0.5*theta_y );
			
			double  r  = exp( - dt*D_theta_y * theta_x*theta_x - dt*D_theta_y * theta_y*theta_y );

			B[i] *= pycuda::complex<double>( r*cos(phase), -r*sin(phase) );

			"""
  		       ,"exp_x_theta_GPU",
			preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants + potentialStringC ) 

		# Ehrenfest theorems .................................................................................

		x_Define    = "\n#define x(i)    dx*( (i%gridDIM_x) - 0.5*gridDIM_x )\n"
		p_x_Define  = "\n#define p_x(i)  dp_x*( ((i/gridDIM_x) % gridDIM_x)-0.5*gridDIM_x)\n"

		y_Define    = "\n#define y(i)   dy  *( (i/(gridDIM_x*gridDIM_x)) % gridDIM_y  - 0.5*gridDIM_y)\n"
		p_y_Define  = "\n#define p_y(i) dp_y*(  i/(gridDIM_x*gridDIM_x*gridDIM_y) - 0.5*gridDIM_y )\n"

		p_x_p_y_Define = p_x_Define + p_y_Define
		phaseSpaceDefine =  p_x_Define + p_y_Define + x_Define + y_Define

		self.Average_x_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( x(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments= "pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES"+x_Define+self.CUDA_constants)

		self.Average_p_x_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( p_x(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments="pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES" +p_x_Define+self.CUDA_constants)

		self.Average_y_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( y(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments= "pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES"+y_Define+self.CUDA_constants)

		self.Average_p_y_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>( p_y(i)*dx*dy*dp_x*dp_y*W[i] )",
        			arguments= "pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES"+p_y_Define+self.CUDA_constants)

		#
		kineticString = self.kineticString.replace( 'p_x' , 'p_x(i)'  )
		kineticString =      kineticString.replace( 'p_y' , 'p_y(i)'  )
		potentialString = (self.potentialString.replace( 'x'   , 'x(i)'    )).replace( 'y'   , 'y(i)'    )
 		energyString = kineticString + "+" + potentialString


		print "\n"
		print energyString

		self.Energy_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "pycuda::real<double>(("+energyString+")*dx*dy*dp_x*dp_y*W[i])",
        			arguments= "pycuda::complex<double> *W",
				preamble = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

	def Gaussian_CPU(x,mu,sigma):
		return np.exp( - (x-mu)**2/sigma**2/2.  )/(sigma*np.sqrt( 2*np.pi  ))

	def fftGaussian_CPU(p,mu,sigma):
		return np.exp(-1j*mu*p , dtype=np.float64 )*np.exp( - p**2*sigma**2/2. , dtype=np.float64 )	

		
	def Norm_GPU( self, W_gpu ):
		norm = gpuarray.sum(W_gpu).get()
		return norm *self.dx * self.dp_x * self.dy * self.dp_y 

	#
	def FAFT_64_128(self, W_gpu):
		print ' FAFT '
		cuda_faft128( int(W_gpu.gpudata),  self.dx,   self.delta_x,    self.FAFT_segment_axes0, self.FAFT_axes0, self.NF  )
		cuda_faft128( int(W_gpu.gpudata),  self.dp_x, self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes1, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dy,   self.delta_y,    self.FAFT_segment_axes2, self.FAFT_axes2, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dp_y, self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes3, self.NF  )

	def iFAFT_64_128(self, W_gpu):
		cuda_faft128( int(W_gpu.gpudata), self.dx,   -self.delta_x,   self.FAFT_segment_axes0, self.FAFT_axes0, self.NF )
		cuda_faft128( int(W_gpu.gpudata), self.dp_x, -self.delta_p_x, self.FAFT_segment_axes1, self.FAFT_axes1, self.NF )
		cuda_faft64(  int(W_gpu.gpudata), self.dy,   -self.delta_y,   self.FAFT_segment_axes2, self.FAFT_axes2, self.NF )
		cuda_faft64(  int(W_gpu.gpudata), self.dp_y, -self.delta_p_y, self.FAFT_segment_axes3, self.FAFT_axes3, self.NF )

	#
	def Fourier_X_To_Lambda_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dp_x, self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes1, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dp_y, self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes2, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_Lambda_To_X_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dp_x, -self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes1, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dp_y, -self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes2, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_P_To_Theta_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dp_x, self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dp_y, self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes3, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_Theta_To_P_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dp_x, -self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dp_y, -self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes3, self.NF  )
		W_gpu /= W_gpu.size

	
	def Run(self):

		try :
			import os
			os.remove (self.fileName)
		except OSError:
			pass


		self.file = h5py.File(self.fileName)

		#self.WriteHDF5_variables()
		#self.file.create_dataset('Hamiltonian', data = self.Hamiltonian.real )


		print '         GPU memory Total       ', pycuda.driver.mem_get_info()[1]/float(2**30) , 'GB'
		print '         GPU memory Free        ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'

		timeRangeIndex = range(0, self.timeSteps+1)

		W_gpu = self.W_init_gpu


		average_x   = []
		average_p_x = []

		average_y   = []
		average_p_y = []
		energy      = []

		for tIndex in timeRangeIndex:

			print ' t index = ', tIndex

			norm   = self.Norm_GPU( W_gpu )
      		        W_gpu /= norm

			average_x.append(   self.Average_x_GPU  (W_gpu).get()  )
			average_p_x.append( self.Average_p_x_GPU(W_gpu).get()  )

			average_y.append(   self.Average_y_GPU  (W_gpu).get()  )
			average_p_y.append( self.Average_p_y_GPU(W_gpu).get()  )
			energy.append(      self.Energy_GPU(W_gpu).get()       )

			# p x  ->  p lambda
			self.Fourier_X_To_Lambda_GPU( W_gpu )

			self.exp_p_lambda_GPU( W_gpu )

			# p lambda  ->  p x
			self.Fourier_Lambda_To_X_GPU( W_gpu )
			#  p x  -> theta x
			self.Fourier_P_To_Theta_GPU( W_gpu )

			self.exp_x_theta_GPU( W_gpu )

			# theta x  -> p x
			self.Fourier_Theta_To_P_GPU( W_gpu )


		norm   = self.Norm_GPU( W_gpu )
      		W_gpu /= norm

		self.average_x   = np.array(average_x  )
		self.average_p_x = np.array(average_p_x)
		self.average_y   = np.array(average_y  )
		self.average_p_y = np.array(average_p_y)
		self.energy      = np.array(energy)

		self.file['/Ehrenfest/energy']       = self.energy
		self.file['/Ehrenfest/average_x']    = self.average_x
		self.file['/Ehrenfest/average_p_x']  = self.average_p_x
		self.file['/Ehrenfest/average_y']    = self.average_y
		self.file['/Ehrenfest/average_p_y']  = self.average_p_y

			
		

		




		











