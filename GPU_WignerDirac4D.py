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

# FAFT 64-points
_faft64_4D = ctypes.cdll.LoadLibrary( DIR_BASE_FAFT+'FAFT64_4D_Z2Z.so' )
_faft64_4D.FAFT64_4D_Z2Z.restype = int
_faft64_4D.FAFT64_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double ]
_faft64_4D.IFAFT64_4D_Z2Z.restype = int
_faft64_4D.IFAFT64_4D_Z2Z.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, 
                                       ctypes.c_int, ctypes.c_int, ctypes.c_double ]

faft64  = _faft64_4D.FAFT64_4D_Z2Z
ifaft64 = _faft64_4D.IFAFT64_4D_Z2Z

#



#..................................................................


class GPU_WignerDirac4D:
	"""
	
	"""
	def __init__( self ):


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

		self.dVolume = self.dp_y*self.dy*self.dp_x*self.dx

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
		self.theta_y_range = np.linspace(-self.theta_y_amplitude, self.theta_y_amplitude -self.dtheta_y, self.gridDIM_y  )#0
		self.lambda_y_range= np.linspace(-self.lambda_y_amplitude,self.lambda_y_amplitude-self.dlambda_y,self.gridDIM_p_y)#1
		self.theta_x_range = np.linspace(-self.theta_x_amplitude, self.theta_x_amplitude -self.dtheta_x, self.gridDIM_x  )#2
		self.lambda_x_range= np.linspace(-self.lambda_x_amplitude,self.lambda_x_amplitude-self.dlambda_x,self.gridDIM_p_x)#3

		# Grid 
		self.p_y = self.p_y_range[ :, np.newaxis, np.newaxis, np.newaxis ]   #axis 0
		self.y   =   self.y_range[ np.newaxis, :, np.newaxis, np.newaxis ]   #axis 1
		self.p_x = self.p_x_range[ np.newaxis, np.newaxis, :, np.newaxis ]   #axis 2
		self.x   =   self.x_range[ np.newaxis, np.newaxis, np.newaxis, : ]   #axis 3
		
		self.CUDA_constants =  '\n'
		self.CUDA_constants += '__device__ double c      = %f;\n'%self.c
		self.CUDA_constants += '__device__ double HBar   = %f;\n'%self.HBar
		self.CUDA_constants += '__device__ double dt   = %f;\n'%self.dt
		self.CUDA_constants += '__device__ double mass = %f;\n'%self.mass

		self.CUDA_constants += '__device__ double dp_y   = %f; \n'%self.dp_y
		self.CUDA_constants += '__device__ double dy     = %f; \n'%self.dy
		self.CUDA_constants += '__device__ double dp_x   = %f; \n'%self.dp_x
		self.CUDA_constants += '__device__ double dx     = %f; \n'%self.dx

		self.CUDA_constants += '__device__ double dtheta_y   = %f; \n'%self.dtheta_y
		self.CUDA_constants += '__device__ double dlambda_y  = %f; \n'%self.dlambda_y
		self.CUDA_constants += '__device__ double dtheta_x   = %f; \n'%self.dtheta_x
		self.CUDA_constants += '__device__ double dlambda_x  = %f; \n'%self.dlambda_x

		self.CUDA_constants += '__device__ int gridDIM_x = %d; \n'%self.gridDIM_x
		self.CUDA_constants += '__device__ int gridDIM_y = %d; \n'%self.gridDIM_y

		try :
			self.CUDA_constants += '__device__ double D_lambda_x = %f; \n'%self.D_lambda_x
			self.CUDA_constants += '__device__ double D_lambda_y = %f; \n'%self.D_lambda_y
			self.CUDA_constants += '__device__ double D_theta_x  = %f; \n'%self.D_theta_x
			self.CUDA_constants += '__device__ double D_theta_y  = %f; \n'%self.D_theta_y
		except AttributeError:
			pass

		self.CUDA_constants +=  '\n'		

		print self.CUDA_constants

		#...........................................................................

		print '         GPU memory Total               ', pycuda.driver.mem_get_info()[1]/float(2**30) , 'GB'
		print '         GPU memory Free  (Before)      ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'

		self.W11  = gpuarray.zeros(
				( self.gridDIM_p_y, self.gridDIM_y, self.gridDIM_p_x, self.gridDIM_x ), dtype=np.complex128 )

		
		self.W12  = gpuarray.zeros_like(self.W11 )
		self.W13  = gpuarray.zeros_like(self.W11 )
		self.W14  = gpuarray.zeros_like(self.W11 )

		self.W21  = gpuarray.zeros_like(self.W11 )
		self.W22  = gpuarray.zeros_like(self.W11 )
		self.W23  = gpuarray.zeros_like(self.W11 )
		self.W24  = gpuarray.zeros_like(self.W11 )

		self.W31  = gpuarray.zeros_like(self.W11 )
		self.W32  = gpuarray.zeros_like(self.W11 )
		self.W33  = gpuarray.zeros_like(self.W11 )
		self.W34  = gpuarray.zeros_like(self.W11 )

		self.W41  = gpuarray.zeros_like(self.W11 )
		self.W42  = gpuarray.zeros_like(self.W11 )
		self.W43  = gpuarray.zeros_like(self.W11 )
		self.W44  = gpuarray.zeros_like(self.W11 )
		
		print '         GPU memory Free  (After)       ', pycuda.driver.mem_get_info()[0]/float(2**30) , 'GB'	

		#............................................................................

		indexUnpack_x_p_string = """
			int i_x   = i%64;
			int i_p_x = (i/64) % 64;
			int i_y   = (i/(64*64)) % 64;
			int i_p_y = i/(64*64*64);

			double x   = dx  *( i_x   - 0.5*gridDIM_x );
			double p_x = dp_x*( i_p_x - 0.5*gridDIM_x );
			double y   = dy  *( i_y   - 0.5*gridDIM_y );
			double p_y = dp_y*( i_p_y - 0.5*gridDIM_y );			
			"""

		indexUnpack_lambda_theta_string = """
			int i_x   = i%64;
			int i_p_x = (i/64) % 64;
			int i_y   = (i/(64*64)) % 64;
			int i_p_y = i/(64*64*64);

			double lambda_x  = dlambda_x * ( i_x   - 0.5*gridDIM_x );
			double theta_x   = dtheta_x  * ( i_p_x - 0.5*gridDIM_x );
			double lambda_y  = dlambda_y * ( i_y   - 0.5*gridDIM_y );
			double theta_y   = dtheta_y  * ( i_p_y - 0.5*gridDIM_y );			
			"""

		indexUnpack_lambda_p_string = """
			int i_x   = i%64;
			int i_p_x = (i/64) % 64;
			int i_y   = (i/(64*64)) % 64;
			int i_p_y = i/(64*64*64);

			double lambda_x   = dlambda_x*( i_x   - 0.5*gridDIM_x );
			double p_x        = dp_x     *( i_p_x - 0.5*gridDIM_x );
			double lambda_y   = dlambda_y*( i_y   - 0.5*gridDIM_y );
			double p_y        = dp_y     *( i_p_y - 0.5*gridDIM_y );			
			"""
		indexUnpack_x_theta_string = """
			int i_x   = i%64;
			int i_p_x = (i/64) % 64;
			int i_y   = (i/(64*64)) % 64;
			int i_p_y = i/(64*64*64);

			double x       = dx      *( i_x   - 0.5*gridDIM_x );
			double theta_x = dtheta_x*( i_p_x - 0.5*gridDIM_x );
			double y       = dy      *( i_y   - 0.5*gridDIM_y );
			double theta_y = dtheta_y*( i_p_y - 0.5*gridDIM_y );	
			"""

		#...............................................................................................		

		self.Real_GPU = ElementwiseKernel(
		    """pycuda::complex<double> *W"""
		    ,
		    """ W[i] = pycuda::real<double>( W[i] ); """
		    ,"Gaussian",  preamble = "#define _USE_MATH_DEFINES"  )	


		#
		self.Gaussian_GPU = ElementwiseKernel(
		    """pycuda::complex<double> *W , 
				double    mu_p_y, double    mu_y, double    mu_p_x, double    mu_x , 
				double sigma_p_y, double sigma_y, double sigma_p_x, double sigma_x """
		    ,
		    indexUnpack_x_p_string + """
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
		   indexUnpack_x_p_string + """
			double temp  = (mass*pow( omega_x*(x-x_mu) ,2) + pow(p_x-p_x_mu,2)/mass)/omega_x;   
			       temp += (mass*pow( omega_y*(y-y_mu) ,2) + pow(p_y-p_y_mu,2)/mass)/omega_y;

			W[i] = pycuda::complex<double>(  exp(-temp) , 0. ); 
					          """
		   ,"Gaussian",  preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants )	


		#....................................................................................................


		def Real_4x4_GPU(self, W11,W22,W33,W44):
			self.Real_GPU(W11)  
			self.Real_GPU(W22)  
			self.Real_GPU(W33)  
			self.Real_GPU(W44)
		

		#

		self.boundary_p_GPU = ElementwiseKernel(
"""
pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44""",
		indexUnpack_x_p_string + """
			double  r  = 1. - exp( -7.*pow(p_x - 0.5*dp_x*(64)   ,4)  );
				r *= 1. - exp( -7.*pow(p_x + 0.5*dp_x*(64-1) ,4)  );
				r *= 1. - exp( -7.*pow(p_y - 0.5*dp_y*(64)   ,4)  );
				r *= 1. - exp( -7.*pow(p_y + 0.5*dp_y*(64-1) ,4)  );
			
			W11[i] = r*W11[i];
			W12[i] = r*W12[i];
			W13[i] = r*W13[i];
			W14[i] = r*W14[i];

			W21[i] = r*W21[i];
			W22[i] = r*W22[i];
			W23[i] = r*W23[i];
			W24[i] = r*W24[i];

			W31[i] = r*W31[i];
			W32[i] = r*W32[i];
			W33[i] = r*W33[i];
			W34[i] = r*W34[i];

			W41[i] = r*W41[i];
			W42[i] = r*W42[i];
			W43[i] = r*W43[i];
			W44[i] = r*W44[i];
					""", 
			"boundary_p_GPU",
			preamble = "#define _USE_MATH_DEFINES\n" + self.CUDA_constants) 

		self.boundary_xp_GPU = ElementwiseKernel(
"""
pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44""",
		indexUnpack_x_p_string + """
			double  r  = 1. - exp( -7.*pow(p_x - 0.5*dp_x*(64)   ,4)  );
				r *= 1. - exp( -7.*pow(p_x + 0.5*dp_x*(64-1) ,4)  );
				r *= 1. - exp( -7.*pow(p_y - 0.5*dp_y*(64)   ,4)  );
				r *= 1. - exp( -7.*pow(p_y + 0.5*dp_y*(64-1) ,4)  );

			r *= 1. - exp( -0.005*pow(x - 0.5*dx*(64)   ,4)  );
			r *= 1. - exp( -0.005*pow(x + 0.5*dx*(64-1) ,4)  );
			r *= 1. - exp( -0.005*pow(y - 0.5*dy*(64)   ,4)  );
			r *= 1. - exp( -0.005*pow(y + 0.5*dy*(64-1) ,4)  );
			
			W11[i] = r*W11[i];
			W12[i] = r*W12[i];
			W13[i] = r*W13[i];
			W14[i] = r*W14[i];

			W21[i] = r*W21[i];
			W22[i] = r*W22[i];
			W23[i] = r*W23[i];
			W24[i] = r*W24[i];

			W31[i] = r*W31[i];
			W32[i] = r*W32[i];
			W33[i] = r*W33[i];
			W34[i] = r*W34[i];

			W41[i] = r*W41[i];
			W42[i] = r*W42[i];
			W43[i] = r*W43[i];
			W44[i] = r*W44[i];
					""", 
			"boundary_xp_GPU",
			preamble = "#define _USE_MATH_DEFINES\n" + self.CUDA_constants) 		



		# ...................................................................................................
		#
		#                     Kinetic propagator         ....................................................
		#
		#....................................................................................................


		kineticStringC = """ 
		__device__ double Omega(double p_x, double p_y, double m){ 
			return  c/HBar*sqrt(pow(m*c,2) + pow(p_x,2) + pow(p_y,2));
		 }"""
  		
		kineticStringC += """
		__device__ pycuda::complex<double> K11(double p1,double p2,double m,double dt){
	return pycuda::complex<double>(cos(dt*Omega(p1,p2,m)),-((pow(c,2)*m*sin(dt*Omega(p1,p2,m)))/(Omega(p1,p2,m)*HBar)));}

		__device__ pycuda::complex<double> K14(double p1,double p2,double m,double dt){ 
	return pycuda::complex<double>(-((c*p2*sin(dt*Omega(p1,p2,m)))/(Omega(p1,p2,m)*HBar)),-((c*p1*sin(dt*Omega(p1,p2,m)))/(Omega(p1,p2,m)*HBar)));}
		"""

		self.exp_p_lambda_GPU = ElementwiseKernel(
"""
pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44"""
			,
			indexUnpack_lambda_p_string + """ 				
 			double p_x_plus   = p_x + 0.5*HBar*lambda_x;
			double p_y_plus   = p_y + 0.5*HBar*lambda_y;
			double p_x_minus  = p_x - 0.5*HBar*lambda_x;
			double p_y_minus  = p_y - 0.5*HBar*lambda_y;

			double  r  = exp( - dt*D_lambda_x*pow(lambda_x,2) - dt*D_lambda_y*pow(lambda_y,2) );

			
			
			double m = mass;
			//double m = mass/2.;

			pycuda::complex<double> KL11 = K11(p_x_plus,p_y_plus, m, dt);
			pycuda::complex<double> KL14 = K14(p_x_plus,p_y_plus, m, dt);  

			pycuda::complex<double> KR11 = K11(p_x_minus,p_y_minus, m,-dt);
			pycuda::complex<double> KR14 = K14(p_x_minus,p_y_minus, m,-dt);  

			pycuda::complex<double> W11_, W12_, W13_, W14_, W21_, W22_, W23_, W24_;
		        pycuda::complex<double> W31_, W32_, W33_, W34_, W41_, W42_, W43_, W44_;

			pycuda::complex<double> W11__, W12__, W13__, W14__, W21__, W22__, W23__, W24__;
		        pycuda::complex<double> W31__, W32__, W33__, W34__, W41__, W42__, W43__, W44__;

			W11_ = W11[i];
			W12_ = W12[i];
			W13_ = W13[i];
			W14_ = W14[i];

			W21_ = W21[i];
			W22_ = W22[i];
			W23_ = W23[i];
			W24_ = W24[i];
		
			W31_ = W31[i];
			W32_ = W32[i];
			W33_ = W33[i];
			W34_ = W34[i];
	
			W41_ = W41[i];
			W42_ = W42[i];
			W43_ = W43[i];
			W44_ = W44[i];

W11__ = KL11*KR11*W11_ - KL11*pycuda::conj<double>(KR14)*W14_ + KL14*KR11*W41_ - KL14*pycuda::conj<double>(KR14)*W44_;
W12__ = KL11*KR11*W12_ + KL11*KR14*W13_ + KL14*KR11*W42_ + KL14*KR14*W43_;
W13__ = -(KL11*pycuda::conj<double>(KR14)*W12_) + KL11*pycuda::conj<double>(KR11)*W13_ - KL14*pycuda::conj<double>(KR14)*W42_ + KL14*pycuda::conj<double>(KR11)*W43_;
W14__ = KL11*KR14*W11_ + KL11*pycuda::conj<double>(KR11)*W14_ + KL14*KR14*W41_ + KL14*pycuda::conj<double>(KR11)*W44_;
W21__ = KL11*KR11*W21_ - KL11*pycuda::conj<double>(KR14)*W24_ - KR11*pycuda::conj<double>(KL14)*W31_ + pycuda::conj<double>(KL14)*pycuda::conj<double>(KR14)*W34_;
W22__ = KL11*KR11*W22_ + KL11*KR14*W23_ - KR11*pycuda::conj<double>(KL14)*W32_ - KR14*pycuda::conj<double>(KL14)*W33_;
W23__ = -(KL11*pycuda::conj<double>(KR14)*W22_) + KL11*pycuda::conj<double>(KR11)*W23_ + pycuda::conj<double>(KL14)*pycuda::conj<double>(KR14)*W32_ - pycuda::conj<double>(KL14)*pycuda::conj<double>(KR11)*W33_;
W24__ = KL11*KR14*W21_ + KL11*pycuda::conj<double>(KR11)*W24_ - KR14*pycuda::conj<double>(KL14)*W31_ - pycuda::conj<double>(KL14)*pycuda::conj<double>(KR11)*W34_;
W31__ = KL14*KR11*W21_ - KL14*pycuda::conj<double>(KR14)*W24_ + KR11*pycuda::conj<double>(KL11)*W31_ - pycuda::conj<double>(KL11)*pycuda::conj<double>(KR14)*W34_;
W32__ = KL14*KR11*W22_ + KL14*KR14*W23_ + KR11*pycuda::conj<double>(KL11)*W32_ + KR14*pycuda::conj<double>(KL11)*W33_;
W33__ = -(KL14*pycuda::conj<double>(KR14)*W22_) + KL14*pycuda::conj<double>(KR11)*W23_ - pycuda::conj<double>(KL11)*pycuda::conj<double>(KR14)*W32_ + pycuda::conj<double>(KL11)*pycuda::conj<double>(KR11)*W33_;
W34__ = KL14*KR14*W21_ + KL14*pycuda::conj<double>(KR11)*W24_ + KR14*pycuda::conj<double>(KL11)*W31_ + pycuda::conj<double>(KL11)*pycuda::conj<double>(KR11)*W34_;
W41__ = -(KR11*pycuda::conj<double>(KL14)*W11_) + pycuda::conj<double>(KL14)*pycuda::conj<double>(KR14)*W14_ + KR11*pycuda::conj<double>(KL11)*W41_ - pycuda::conj<double>(KL11)*pycuda::conj<double>(KR14)*W44_;
W42__ = -(KR11*pycuda::conj<double>(KL14)*W12_) - KR14*pycuda::conj<double>(KL14)*W13_ + KR11*pycuda::conj<double>(KL11)*W42_ + KR14*pycuda::conj<double>(KL11)*W43_;
W43__ = pycuda::conj<double>(KL14)*pycuda::conj<double>(KR14)*W12_ - pycuda::conj<double>(KL14)*pycuda::conj<double>(KR11)*W13_ - pycuda::conj<double>(KL11)*pycuda::conj<double>(KR14)*W42_ + pycuda::conj<double>(KL11)*pycuda::conj<double>(KR11)*W43_;
W44__ = -(KR14*pycuda::conj<double>(KL14)*W11_) - pycuda::conj<double>(KL14)*pycuda::conj<double>(KR11)*W14_ + KR14*pycuda::conj<double>(KL11)*W41_ + pycuda::conj<double>(KL11)*pycuda::conj<double>(KR11)*W44_;

			W11[i] = r*W11__;
			W12[i] = r*W12__;
			W13[i] = r*W13__;
			W14[i] = r*W14__;

			W21[i] = r*W21__;
			W22[i] = r*W22__;
			W23[i] = r*W23__;
			W24[i] = r*W24__;

			W31[i] = r*W31__;
			W32[i] = r*W32__;
			W33[i] = r*W33__;
			W34[i] = r*W34__;

			W41[i] = r*W41__;
			W42[i] = r*W42__;
			W43[i] = r*W43__;
			W44[i] = r*W44__;

			"""
  		       ,"exp_p_lambda_GPU",
			preamble = "#define _USE_MATH_DEFINES\n" + self.CUDA_constants +kineticStringC ) 



		# ...................................................................................................
		#
		#                     Potential propagator       ....................................................
		#
		#....................................................................................................


		Potential_Propagator_source = """
//
//   
#include <pycuda-complex.hpp>
#include<math.h>
#define _USE_MATH_DEFINES"""+self.CUDA_constants+"""
__device__  double A0(double t, double x, double y)
{
    return """+self.potential_0_String+""";
}
__device__  double A1(double t, double x, double y)
{
   return """+self.potential_1_String+""" ;
}
__device__  double A2(double t, double x, double y)
{
   return """+self.potential_2_String+""" ;
}

__device__  double A3(double t, double x, double y)
{
   return """+self.potential_3_String+""" ;
}

//............................................................................................................
__global__ void Potential_Propagator_Kernel(
   pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14,
   pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
   pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
   pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44 )
{
  
  int i = threadIdx.x + blockIdx.x*(64) + blockIdx.y*(64*64) + blockIdx.z*(64*64*64);
	
  double x       =       dx*( double(threadIdx.x) - 0.5*gridDIM_x  );
  double theta_x = dtheta_x*( double(blockIdx.x)  - 0.5*gridDIM_x  );
  double y       =       dy*( double(blockIdx.y)  - 0.5*gridDIM_y  );
  double theta_y = dtheta_y*( double(blockIdx.z)  - 0.5*gridDIM_y  );
  double x_minus = x - 0.5*theta_x;
  double x_plus  = x + 0.5*theta_x;
 
  double y_minus = y - 0.5*theta_y;
  double y_plus  = y + 0.5*theta_y;
  double t=0.;
  double F;
  //double m = mass/2.;
  double m = 1e-6;
	
  pycuda::complex<double>  U11,      U13, U14;
  pycuda::complex<double>       U22, U23, U24;
  pycuda::complex<double>  U31,	U32, U33     ;
  pycuda::complex<double>  U41, U42,      U44;
  //..........................................................................
  pycuda::complex<double> W11_, W12_, W13_, W14_;
  pycuda::complex<double> W21_, W22_, W23_, W24_;
  pycuda::complex<double> W31_, W32_, W33_, W34_;
  pycuda::complex<double> W41_, W42_, W43_, W44_;
  pycuda::complex<double> W11__, W12__, W13__, W14__;
  pycuda::complex<double> W21__, W22__, W23__, W24__;
  pycuda::complex<double> W31__, W32__, W33__, W34__;
  pycuda::complex<double> W41__, W42__, W43__, W44__;  
 
 //-----------------------------------------------------------------------------	
  W11_ = W11[i];
  W12_ = W12[i];
  W13_ = W13[i];
  W14_ = W14[i];
  W21_ = W21[i];
  W22_ = W22[i];
  W23_ = W23[i];
  W24_ = W24[i];
  W31_ = W31[i];
  W32_ = W32[i];
  W33_ = W33[i];
  W34_ = W34[i];
  W41_ = W41[i];
  W42_ = W42[i];
  W43_ = W43[i];
  W44_ = W44[i];
 //------------------------------------------------------------------------------- 
 //.............................
 //                 UW
 //.............................
  
  F = sqrt( pow(m*c,2) + pow(A1(0.,x_minus,y_minus),2) + pow(A2(0.,x_minus,y_minus),2)); 
  U11 = pycuda::complex<double>( cos(c*dt*F/HBar) , -m*c*sin(-c*dt*F/HBar)/F );
  U22 = pycuda::complex<double>( cos(c*dt*F/HBar) , -m*c*sin(-c*dt*F/HBar)/F );
  U33 = pycuda::complex<double>( cos(c*dt*F/HBar) ,  m*c*sin(-c*dt*F/HBar)/F );
  U44 = pycuda::complex<double>( cos(c*dt*F/HBar) ,  m*c*sin(-c*dt*F/HBar)/F );
 
  U14 = pycuda::complex<double>(-A2(0.,x_minus,y_minus), -A1(0.,x_minus,y_minus)  )*sin(-c*dt*F/HBar)/F;
  U41 = pycuda::complex<double>( A2(0.,x_minus,y_minus), -A1(0.,x_minus,y_minus)  )*sin(-c*dt*F/HBar)/F;
  U23 = pycuda::complex<double>( A2(0.,x_minus,y_minus), -A1(0.,x_minus,y_minus)  )*sin(-c*dt*F/HBar)/F;
  U32 = pycuda::complex<double>(-A2(0.,x_minus,y_minus), -A1(0.,x_minus,y_minus)  )*sin(-c*dt*F/HBar)/F;
  
  //U13 = 0.;
  //U24 = 0.;
  //U31 = 0.;
  //U42 = 0.;
  U13 = pycuda::complex<double>( 0. ,  A3(0.,x_minus,y_minus ) *sin(c*dt*F/HBar)/F  );
  U24 = pycuda::complex<double>( 0. , -A3(0.,x_minus,y_minus ) *sin(c*dt*F/HBar)/F  );
  U31 = pycuda::complex<double>( 0. ,  A3(0.,x_minus,y_minus ) *sin(c*dt*F/HBar)/F  );
  U42 = pycuda::complex<double>( 0. , -A3(0.,x_minus,y_minus ) *sin(c*dt*F/HBar)/F  );
  
  W11__ =  U11*W11_               +  U13*W31_	+  U14*W41_;
  W21__ =  	 	U22*W21_  +  U23*W31_   +  U24*W41_;
  W31__ =  U31*W11_ +	U32*W21_  +  U33*W31_   	   ;
  W41__ =  U41*W11_ +   U42*W21_  		+  U44*W41_;


  W12__ =  U11*W12_               +  U13*W32_	+  U14*W42_;
  W22__ =  		U22*W22_  +  U23*W32_   +  U24*W42_;
  W32__ =  U31*W12_ +	U32*W22_  +  U33*W32_   	   ;
  W42__ =  U41*W12_ + 	U42*W22_  		+  U44*W42_;


  W13__ =  U11*W13_               +  U13*W33_	+  U14*W43_;
  W23__ =  		U22*W23_  +  U23*W33_   +  U24*W43_;
  W33__ =  U31*W13_ +	U32*W23_  +  U33*W33_   	   ;
  W43__ =  U41*W13_ +   U42*W23_  		+  U44*W43_;


  W14__ =  U11*W14_               +  U13*W34_	+  U14*W44_;
  W24__ =  		U22*W24_  +  U23*W34_   +  U24*W44_;
  W34__ =  U31*W14_ +	U32*W24_  +  U33*W34_   	   ;
  W44__ =  U41*W14_ +   U42*W24_  		+  U44*W44_;

 //...............................
 //        WU
 //...............................

  F = sqrt( pow(m*c,2) + pow(A1(0.,x_plus,y_plus),2) + pow(A2(0.,x_plus,y_plus),2)); 

  U11 = pycuda::complex<double>( cos(c*dt*F/HBar) ,  m*c*sin(-c*dt*F/HBar)/F );
  U22 = pycuda::complex<double>( cos(c*dt*F/HBar) ,  m*c*sin(-c*dt*F/HBar)/F );
  U33 = pycuda::complex<double>( cos(c*dt*F/HBar) , -m*c*sin(-c*dt*F/HBar)/F );
  U44 = pycuda::complex<double>( cos(c*dt*F/HBar) , -m*c*sin(-c*dt*F/HBar)/F );
  U14 = pycuda::complex<double>(-A2(0.,x_plus,y_plus), -A1(0.,x_plus,y_plus)  )*sin(c*dt*F/HBar)/F;
  U41 = pycuda::complex<double>( A2(0.,x_plus,y_plus), -A1(0.,x_plus,y_plus)  )*sin(c*dt*F/HBar)/F;
  U23 = pycuda::complex<double>( A2(0.,x_plus,y_plus), -A1(0.,x_plus,y_plus)  )*sin(c*dt*F/HBar)/F;
  U32 = pycuda::complex<double>(-A2(0.,x_plus,y_plus), -A1(0.,x_plus,y_plus)  )*sin(c*dt*F/HBar)/F;

  U13 = pycuda::complex<double>( 0. ,  A3(0.,x_plus,y_plus ) *sin(-c*dt*F/HBar)/F  );
  U24 = pycuda::complex<double>( 0. , -A3(0.,x_plus,y_plus ) *sin(-c*dt*F/HBar)/F  );
  U31 = pycuda::complex<double>( 0. ,  A3(0.,x_plus,y_plus ) *sin(-c*dt*F/HBar)/F  );
  U42 = pycuda::complex<double>( 0. , -A3(0.,x_plus,y_plus ) *sin(-c*dt*F/HBar)/F  );

  W11_ =  W11__*U11  		    +  W13__*U31  +  W14__*U41;
  W21_ =  W21__*U11  		    +  W23__*U31  +  W24__*U41;
  W31_ =  W31__*U11  	 	    +  W33__*U31  +  W34__*U41;
  W41_ =  W41__*U11  	 	    +  W43__*U31  +  W44__*U41;

  W12_ =   		 W12__*U22  +  W13__*U32  +  W14__*U42;
  W22_ =    		 W22__*U22  +  W23__*U32  +  W24__*U42;
  W32_ =    		 W32__*U22  +  W33__*U32  +  W34__*U42;
  W42_ =    		 W42__*U22  +  W43__*U32  +  W44__*U42;

  W13_ =  W11__*U13	+W12__*U23  +  W13__*U33  	      ;
  W23_ =  W21__*U13	+W22__*U23  +  W23__*U33  	      ;
  W33_ =  W31__*U13	+W32__*U23  +  W33__*U33  	      ;
  W43_ =  W41__*U13 	+W42__*U23  +  W43__*U33  	      ;

  W14_ =  W11__*U14	+W12__*U24  		  +  W14__*U44;
  W24_ =  W21__*U14  	+W22__*U24   		  +  W24__*U44;
  W34_ =  W31__*U14  	+W32__*U24   		  +  W34__*U44;
  W44_ =  W41__*U14  	+W42__*U24   		  +  W44__*U44;


 //-------------------------------------------------------------------------------
  double phase = A0(t,x_minus,y_minus) - A0(t,x_plus,y_plus);
  pycuda::complex<double> expV = pycuda::complex<double>( cos(dt*phase/HBar) , -sin(dt*phase/HBar)  );	
  W11[i] = expV* W11_;
  W12[i] = expV* W12_;
  W13[i] = expV* W13_;
  W14[i] = expV* W14_; 	
  
  W21[i] = expV* W21_;
  W22[i] = expV* W22_;
  W23[i] = expV* W23_;
  W24[i] = expV* W24_; 
  W31[i] = expV* W31_;
  W32[i] = expV* W32_;
  W33[i] = expV* W33_;
  W34[i] = expV* W34_; 
  W41[i] = expV* W41_;
  W42[i] = expV* W42_;
  W43[i] = expV* W43_;
  W44[i] = expV* W44_;  
  
}
"""

		self.Potential_Propagator=SourceModule(Potential_Propagator_source).get_function("Potential_Propagator_Kernel")
		#......................................................................................................
		#
		#               Initial states 
		#
		#......................................................................................................

		gaussianPsi_ParticleUp = """\n

	__device__ double u0(double u_x,double u_y){ return sqrt( pow(u_x,2) + pow(u_y,2) + c*c ); }

	__device__ pycuda::complex<double> psi1( double x, double y , double u_x, double u_y, double x_sigma, double y_sigma)
	{
		double u0_ = u0(u_x,u_y);
		double phase = mass*(x*u_x + y*u_y);
		double r = exp( -0.5*pow(x/x_sigma,2) -0.5*pow(y/y_sigma,2)  ) * (c + u0_);
		return pycuda::complex<double>( r*cos(phase) , r*sin(phase) );
		}\n


	__device__ pycuda::complex<double> psi4( double x, double y , double u_x, double u_y, double x_sigma, double y_sigma){
		double phase = mass*(x*u_x + y*u_y);
		double r = exp( -0.5*pow(x/x_sigma,2) -0.5*pow(y/y_sigma,2)  );
		return r*pycuda::complex<double>( cos(phase)*u_x - sin(phase)*u_y , sin(phase)*u_x + cos(phase)*u_y ) ;
		}	

	//...................................Majorana...............................

	__device__ pycuda::complex<double> psi_majorana_1(double x,double y,double u_x,double u_y,double x_sigma,double y_sigma){
		return psi1(x,y,u_x,u_y,x_sigma,y_sigma) -  pycuda::conj<double>( psi4(x,y,u_x,u_y,x_sigma,y_sigma) );
	}

	__device__ pycuda::complex<double> psi_majorana_4(double x,double y,double u_x,double u_y,double x_sigma,double y_sigma){
		return psi4(x,y,u_x,u_y,x_sigma,y_sigma) -  pycuda::conj<double>( psi1(x,y,u_x,u_y,x_sigma,y_sigma) );
	}
	\n
	 """

		self.WignerDiracGaussian_ParticleUp_GPU = ElementwiseKernel("""
pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44,
		double u_x, double u_y, double x_mu, double y_mu, double x_sigma, double y_sigma"""
		,
		indexUnpack_x_theta_string + """
		double x_minus = x - 0.5*HBar*theta_x - x_mu;
		double y_minus = y - 0.5*HBar*theta_y - y_mu;

		double x_plus  = x + 0.5*HBar*theta_x - x_mu;
		double y_plus  = y + 0.5*HBar*theta_y - y_mu;
 
		W11[i] =                        psi1(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(  psi1(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma )    );

		W14[i] =                        psi1(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(  psi4(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma )    );	
	 
		W41[i] =                        psi4(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(  psi1(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma )    );		

		W44[i] =                        psi4(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(  psi4(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma )    );


		 """,
		"WignerDiracGaussian_ParticleUp",
		preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants + gaussianPsi_ParticleUp )


		#.........................................................................................
		#				Landao Level
		#.........................................................................................

		landaoPsi = """\n
	__device__ pycuda::complex<double> psi1( double x, double y, double B)
	{
	return pycuda::complex<double>(  exp( -B*0.25*pow(x,2) -B*0.25*pow(y,2)  ) , 0. );
	}\n	
	 	"""
		self.WignerLandaoGround_GPU = ElementwiseKernel("""
pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44,
		double B, double x_offset, double y_offset"""
		,
		indexUnpack_x_theta_string + """
		double x_minus = x - 0.5*HBar*theta_x - x_offset;
		double y_minus = y - 0.5*HBar*theta_y - y_offset;

		double x_plus  = x + 0.5*HBar*theta_x - x_offset;
		double y_plus  = y + 0.5*HBar*theta_y - y_offset;
 
		W11[i] =  psi1(x_minus,y_minus, B )*pycuda::conj<double>(  psi1(x_plus,y_plus,B) );

		 """,
		"WignerDiracGaussian_ParticleUp",
		preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants + landaoPsi )

		#............................... Majorana state ..........................................

		self.WignerDiracGaussian_Majorana_GPU = ElementwiseKernel("""
pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44,
		double u_x, double u_y, double x_mu, double y_mu, double x_sigma, double y_sigma, int MajoranaSign"""
		,
		indexUnpack_x_theta_string + """
		double x_minus = x - 0.5*HBar*theta_x - x_mu;
		double y_minus = y - 0.5*HBar*theta_y - y_mu;
		double x_plus  = x + 0.5*HBar*theta_x - x_mu;
		double y_plus  = y + 0.5*HBar*theta_y - y_mu;
 
		

		W11[i] =                      psi_majorana_1(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(psi_majorana_1(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma ));

		W14[i] =                      psi_majorana_1(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(psi_majorana_4(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma ));

		W41[i] =                      psi_majorana_4(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(psi_majorana_1(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma ));	
	 
		W44[i] =                      psi_majorana_4(x_minus,y_minus, u_x,u_y, x_sigma,y_sigma )* 
			 pycuda::conj<double>(psi_majorana_4(x_plus,y_plus,   u_x,u_y, x_sigma,y_sigma ));

		 """,
		"WignerDiracGaussian_MajoranaPlus",
		preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants + gaussianPsi_ParticleUp )


		#.....................................................................................................
		#
		#                             Particle Projector
		#.....................................................................................................

		projectorFunctions = """
			__device__ double K(double p_x,double p_y, double m){
				return c*sqrt(  pow(m*c,2) + pow(p_x,2) + pow(p_y,2) );
			}

			__device__ double P11 (double p_x, double p_y, double m, int s){
				return 1. + (pow(c,2)*m*s)/K(p_x,p_y,m);
				}

			__device__ double P33 (double p_x, double p_y, double m, int s){
				return 1. - (pow(c,2)*m*s)/K(p_x,p_y,m);
				}

			__device__ pycuda::complex<double> P14 (double p_x, double p_y, double m, int s){
				return pycuda::complex<double>( s*(c*p_x) , s*(-c*p_y) )/K(p_x,p_y,m);
				}
			"""

		self.ParticleProjector_GPU = ElementwiseKernel("""
pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44,
		int sign"""
		,
		indexUnpack_lambda_p_string + """
		double p_x_minus  = p_x - 0.5*HBar*lambda_x;
		double p_y_minus  = p_y - 0.5*HBar*lambda_y;

		double p_x_plus   = p_x + 0.5*HBar*lambda_x;
		double p_y_plus   = p_y + 0.5*HBar*lambda_y;		

		pycuda::complex<double> p14;

		pycuda::complex<double> W11_, W12_, W13_, W14_, W21_, W22_, W23_, W24_;
		pycuda::complex<double> W31_, W32_, W33_, W34_, W41_, W42_, W43_, W44_;

		pycuda::complex<double> W11__, W12__, W13__, W14__, W21__, W22__, W23__, W24__;
		pycuda::complex<double> W31__, W32__, W33__, W34__, W41__, W42__, W43__, W44__;

		W11_ = W11[i]/4.;
		W12_ = W12[i]/4.; 
		W13_ = W13[i]/4.; 
		W14_ = W14[i]/4.;

		W21_ = W21[i]/4.;
		W22_ = W22[i]/4.;
		W23_ = W23[i]/4.;
		W24_ = W24[i]/4.;

		W31_ = W31[i]/4.;
		W32_ = W32[i]/4.;
		W33_ = W33[i]/4.;
		W34_ = W34[i]/4.;

		W41_ = W41[i]/4.;
		W42_ = W42[i]/4.;
		W43_ = W43[i]/4.;
		W44_ = W44[i]/4.;

		double 				pL11 = P11( p_x_plus, p_y_plus, mass, sign);
		double 				pL33 = P33( p_x_plus, p_y_plus, mass, sign);
		pycuda::complex<double>         pL14 = P14( p_x_plus, p_y_plus, mass, sign);

		double  		 	pR11 = P11( p_x_minus, p_y_minus, mass, sign);
		double 			   	pR33 = P33( p_x_minus, p_y_minus, mass, sign);
		pycuda::complex<double>  	pR14 = P14( p_x_minus, p_y_minus, mass, sign);

W11__ = pL11*pR11*W11_ + pL11*pycuda::conj<double>(pR14)*W14_ + pL14*pR11*W41_ + pL14*pycuda::conj<double>(pR14)*W44_;
W12__ = pL11*pR11*W12_ + pL11*pR14*W13_ + pL14*pR11*W42_ + pL14*pR14*W43_;
W13__ = pL11*pycuda::conj<double>(pR14)*W12_ + pL11*pR33*W13_ + pL14*pycuda::conj<double>(pR14)*W42_ + pL14*pR33*W43_;
W14__ = pL11*pR14*W11_ + pL11*pR33*W14_ + pL14*pR14*W41_ + pL14*pR33*W44_;
W21__ = pL11*pR11*W21_ + pL11*pycuda::conj<double>(pR14)*W24_ + pR11*pycuda::conj<double>(pL14)*W31_ + pycuda::conj<double>(pL14)*pycuda::conj<double>(pR14)*W34_;
W22__ = pL11*pR11*W22_ + pL11*pR14*W23_ + pR11*pycuda::conj<double>(pL14)*W32_ + pR14*pycuda::conj<double>(pL14)*W33_;
W23__ = pL11*pycuda::conj<double>(pR14)*W22_ + pL11*pR33*W23_ + pycuda::conj<double>(pL14)*pycuda::conj<double>(pR14)*W32_ + pR33*pycuda::conj<double>(pL14)*W33_;
W24__ = pL11*pR14*W21_ + pL11*pR33*W24_ + pR14*pycuda::conj<double>(pL14)*W31_ + pR33*pycuda::conj<double>(pL14)*W34_;
W31__ = pL14*pR11*W21_ + pL14*pycuda::conj<double>(pR14)*W24_ + pL33*pR11*W31_ + pL33*pycuda::conj<double>(pR14)*W34_;
W32__ = pL14*pR11*W22_ + pL14*pR14*W23_ + pL33*pR11*W32_ + pL33*pR14*W33_;
W33__ = pL14*pycuda::conj<double>(pR14)*W22_ + pL14*pR33*W23_ + pL33*pycuda::conj<double>(pR14)*W32_ + pL33*pR33*W33_;
W34__ = pL14*pR14*W21_ + pL14*pR33*W24_ + pL33*pR14*W31_ + pL33*pR33*W34_;
W41__ = pR11*pycuda::conj<double>(pL14)*W11_ + pycuda::conj<double>(pL14)*pycuda::conj<double>(pR14)*W14_ + pL33*pR11*W41_ + pL33*pycuda::conj<double>(pR14)*W44_;
W42__ = pR11*pycuda::conj<double>(pL14)*W12_ + pR14*pycuda::conj<double>(pL14)*W13_ + pL33*pR11*W42_ + pL33*pR14*W43_;
W43__ = pycuda::conj<double>(pL14)*pycuda::conj<double>(pR14)*W12_ + pR33*pycuda::conj<double>(pL14)*W13_ + pL33*pycuda::conj<double>(pR14)*W42_ + pL33*pR33*W43_;
W44__ = pR14*pycuda::conj<double>(pL14)*W11_ + pR33*pycuda::conj<double>(pL14)*W14_ + pL33*pR14*W41_ + pL33*pR33*W44_;

		
		W11[i] = W11__;
		W12[i] = W12__;
		W13[i] = W13__;
		W14[i] = W14__;

		W21[i] = W21__;
		W22[i] = W22__;
		W23[i] = W23__;
		W24[i] = W24__;

		W31[i] = W31__;
		W32[i] = W32__;
		W33[i] = W33__;
		W34[i] = W34__;

		W41[i] = W41__;
		W42[i] = W42__;
		W43[i] = W43__;
		W44[i] = W44__;

		 """,
		"WignerDiracGaussian_ParticleUp",
		preamble = "#define _USE_MATH_DEFINES" + self.CUDA_constants + projectorFunctions )

		#.....................................................................................................
		#
		# 					Reduction Kernels 
		#
		# Ehrenfest theorems .................................................................................

		volume_Define = "\n#define dV  dx*dy*dp_x*dp_y \n "

		#x_Define    = "\n#define x(i)    dx*( (i%gridDIM_x) - 0.5*gridDIM_x )\n"
		#p_x_Define  = "\n#define p_x(i)  dp_x*( ((i/gridDIM_x) % gridDIM_x)-0.5*gridDIM_x)\n"
		#y_Define    = "\n#define y(i)   dy  *( (i/(gridDIM_x*gridDIM_x)) % gridDIM_y  - 0.5*gridDIM_y)\n"
		#p_y_Define  = "\n#define p_y(i) dp_y*(  i/(gridDIM_x*gridDIM_x*gridDIM_y) - 0.5*gridDIM_y )\n"

		x_Define    = "\n#define x(i)      dx*(  (i%64)             -32.)\n"
		p_x_Define  = "\n#define p_x(i)  dp_x*( ((i/64) % 64)       -32.)\n"

		y_Define    = "\n#define y(i)     dy*(  ((i/(64*64)) % 64)  -32.)\n"
		p_y_Define  = "\n#define p_y(i) dp_y*(  i/(64*64*64)        -32.)\n"

		phaseSpaceDefine =  p_x_Define + p_y_Define + x_Define + y_Define + volume_Define

		diagonalW_string = """pycuda::complex<double> *W11,pycuda::complex<double> *W22,
				      pycuda::complex<double> *W33,pycuda::complex<double> *W44"""

		self.Average_x_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "dV*x(i)*pycuda::real<double>(W11[i]+W22[i]+W33[i]+W44[i])",
        			arguments= diagonalW_string,
				preamble = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		self.Average_x_square_GPU = reduction.ReductionKernel( np.float64, neutral="0",
	       			reduce_expr="a+b", 
				map_expr = "dV*pow(x(i),2)*pycuda::real<double>(W11[i]+W22[i]+W33[i]+W44[i])",
	     			arguments= diagonalW_string,
				preamble = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		self.Average_p_x_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "dV*p_x(i)*pycuda::real<double>(W11[i]+W22[i]+W33[i]+W44[i])",
        			arguments= diagonalW_string,
				preamble = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		self.Average_p_x_square_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = """dV*p_x(i)*p_x(i)*pycuda::real<double>(W11[i]+W22[i]+W33[i]+W44[i])""",
        			arguments= diagonalW_string,
				preamble = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		self.Average_y_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "dV*y(i)*pycuda::real<double>(W11[i]+W22[i]+W33[i]+W44[i])",
        			arguments= diagonalW_string,
				preamble = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		self.Average_p_y_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "dV*p_y(i)*pycuda::real<double>(W11[i]+W22[i]+W33[i]+W44[i])",
        			arguments= diagonalW_string,
				preamble = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		self.Average_y_square_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "dV*y(i)*y(i)*pycuda::real<double>(W11[i]+W22[i]+W33[i]+W44[i])",
        			arguments= diagonalW_string,
				preamble = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		self.Average_p_y_square_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr = "dV*p_y(i)*p_y(i)*pycuda::real<double>(W11[i]+W22[i]+W33[i]+W44[i])",
        			arguments= diagonalW_string,
				preamble = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)


		argumentWString = """
pycuda::complex<double> *W11, pycuda::complex<double> *W12, pycuda::complex<double> *W13, pycuda::complex<double> *W14, 
pycuda::complex<double> *W21, pycuda::complex<double> *W22, pycuda::complex<double> *W23, pycuda::complex<double> *W24,
pycuda::complex<double> *W31, pycuda::complex<double> *W32, pycuda::complex<double> *W33, pycuda::complex<double> *W34,
pycuda::complex<double> *W41, pycuda::complex<double> *W42, pycuda::complex<double> *W43, pycuda::complex<double> *W44"""

		self.Average_Alpha_1_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        		reduce_expr="a+b", 
			map_expr  = "dV*pycuda::real<double>( W14[i] + W41[i] + W23[i] + W32[i])",
   arguments="pycuda::complex<double> *W14,pycuda::complex<double> *W41,pycuda::complex<double> *W23, pycuda::complex<double> *W32",
			preamble  = "#define _USE_MATH_DEFINES" +volume_Define+self.CUDA_constants)

		self.Average_Alpha_2_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr  = "-dV*pycuda::imag<double>( W14[i]-W41[i] + W32[i]-W23[i] )",
   arguments="pycuda::complex<double> *W14,pycuda::complex<double> *W41,pycuda::complex<double> *W23, pycuda::complex<double> *W32",
				preamble  = "#define _USE_MATH_DEFINES" +volume_Define+self.CUDA_constants)


		self.Average_Alpha_1_x_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
			map_expr  = "dV*x(i)*pycuda::real<double>( W14[i] + W41[i] + W23[i] + W32[i])",
   arguments="pycuda::complex<double> *W14,pycuda::complex<double> *W41,pycuda::complex<double> *W23, pycuda::complex<double> *W32",
				preamble  = "#define _USE_MATH_DEFINES" +phaseSpaceDefine+self.CUDA_constants)	

		self.Average_Alpha_2_y_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr  = "-dV*y(i)*pycuda::imag<double>( W14[i]-W41[i] + W32[i]-W23[i] )",
   arguments="pycuda::complex<double> *W14,pycuda::complex<double> *W41,pycuda::complex<double> *W23, pycuda::complex<double> *W32",
				preamble  = "#define _USE_MATH_DEFINES" +phaseSpaceDefine+self.CUDA_constants)

		# ........................................
		#
		# 		Energy
		#
		#.........................................  Kinetic

		energyKineticString =  "c*dV*p_x(i)*pycuda::real<double>(W14[i]+W41[i]+W32[i]+W23[i]) + "

		energyKineticString +=  "-c*dV*p_y(i)*pycuda::imag<double>(W14[i]-W41[i]+W32[i]-W23[i])  + "

		energyKineticString += " dV*mass*c*c*pycuda::real<double>(W11[i]+W22[i]-W33[i]-W44[i]) "


		energyFunctionArguments = """pycuda::complex<double> *W11, pycuda::complex<double> *W22,
					     pycuda::complex<double> *W33, pycuda::complex<double> *W44, 
					     pycuda::complex<double> *W14, pycuda::complex<double> *W41,
					     pycuda::complex<double> *W23, pycuda::complex<double> *W32 """

		self.Average_EnergyKinetic_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr  = energyKineticString,
        			arguments = argumentWString,
				preamble  = "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		self.Average_Energy_Mass_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr = "a+b", 
				map_expr  =   "dV*mass*c*c*pycuda::real<double>(W11[i]+W22[i]-W33[i]-W44[i])",
        			arguments =   argumentWString,
				preamble  =   "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		self.Average_Energy_p_x_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr = "a+b", 
				map_expr  =   "c*dV*p_x(i)*pycuda::real<double>(W14[i]+W41[i]+W32[i]+W23[i])",
        			arguments =   argumentWString,
				preamble  =   "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		self.Average_Energy_p_y_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr = "a+b", 
				map_expr  =   "-c*dV*p_y(i)*pycuda::imag<double>(W14[i]-W41[i]+W32[i]-W23[i])",
        			arguments =   argumentWString,
				preamble  =   "#define _USE_MATH_DEFINES"+phaseSpaceDefine+self.CUDA_constants)

		#....................................... Potential

		potential012_Define  = "\n#define A0  ("+self.potential_0_String+") \n "
		potential012_Define += "\n#define A1  ("+self.potential_1_String+") \n "
		potential012_Define += "\n#define A2  ("+self.potential_2_String+") \n "
		potential012_Define += "\n#define A3  ("+self.potential_3_String+") \n "
		potential012_Define += "\n#define D_1_A_0  ("+self.D_1_potential_0_String+") \n "
		potential012_Define += "\n#define D_1_A_1  ("+self.D_1_potential_1_String+") \n "
		potential012_Define += "\n#define D_1_A_2  ("+self.D_1_potential_2_String+") \n "
		potential012_Define = potential012_Define.replace("x","x(i)").replace("y","y(i)")

		print 'potential012_Define = ',potential012_Define

		energyPotentialString  = "  dV*A1*c*pycuda::real<double>( -W14[i]-W23[i] -W32[i]-W41[i] )"
		energyPotentialString += "- dV*A2*c*pycuda::imag<double>( -W14[i]+W23[i] -W32[i]+W41[i] )"
		energyPotentialString += "+ dV*A3*c*pycuda::real<double>( -W13[i]+W24[i] -W31[i]+W42[i] )"
		energyPotentialString += "+ c*A0*dV*pycuda::real<double>(  W11[i]+W22[i] +W33[i]+W44[i] )"

		potentialDefines = phaseSpaceDefine + self.CUDA_constants + potential012_Define

		energyFunction_plus_time_Arguments = argumentWString+""",double t"""

		self.Average_EnergyPotential_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr  = energyPotentialString,
        			arguments = energyFunction_plus_time_Arguments,
				preamble  = "#define _USE_MATH_DEFINES\n"+potentialDefines)

		#.......................................

		self.Average_D_1_A_0_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr  =  "c*D_1_A_0*dV*pycuda::real<double>( W11[i]+W22[i]+W33[i]+W44[i] )",
        			arguments = energyFunction_plus_time_Arguments,
				preamble  = "#define _USE_MATH_DEFINES\n"+potentialDefines)

		self.Average_D_1_A_1_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr  =  "c* D_1_A_1 *dV*pycuda::real<double>(  W14[i]+W41[i] +W23[i]+W32[i] )",
        			arguments = energyFunction_plus_time_Arguments,
				preamble  = "#define _USE_MATH_DEFINES\n"+potentialDefines)

		self.Average_D_1_A_2_GPU = reduction.ReductionKernel( np.float64, neutral="0",
        			reduce_expr="a+b", 
				map_expr  =  "c* D_1_A_2 *dV*pycuda::real<double>(  W14[i]-W41[i] -W23[i]+W32[i] )",
        			arguments = energyFunction_plus_time_Arguments,
				preamble  = "#define _USE_MATH_DEFINES\n"+potentialDefines)


	#-----------------------------------------------------------------------------------------------------
	#-----------------------------------------------------------------------------------------------------	

	def Gaussian_CPU(x,mu,sigma):
		return np.exp( - (x-mu)**2/sigma**2/2.  )/(sigma*np.sqrt( 2*np.pi  ))

	def fftGaussian_CPU(p,mu,sigma):
		return np.exp(-1j*mu*p , dtype=np.float64 )*np.exp( - p**2*sigma**2/2. , dtype=np.float64 )	

		
	def Norm_GPU( self, W_gpu ):
		norm = gpuarray.sum(W_gpu).get()
		return norm *self.dx * self.dp_x * self.dy * self.dp_y 

	def Norm_4x4_GPU( self, W11, W22, W33, W44 ):
		norm  = gpuarray.sum(W11).get()
		norm += gpuarray.sum(W22).get()
		norm += gpuarray.sum(W33).get()
		norm += gpuarray.sum(W44).get()
		return norm *self.dx * self.dp_x * self.dy * self.dp_y 

	def Normalize_4x4_GPU(self, W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44):
		norm = self.Norm_4x4_GPU( W11, W22, W33, W44 )

		print '			 Norm = ', norm
		W11 /= norm 
		W12 /= norm
		W13 /= norm
		W14 /= norm
 
		W21 /= norm
		W22 /= norm
		W23 /= norm
		W24 /= norm

		W31 /= norm
		W32 /= norm
		W33 /= norm
		W34 /= norm

		W41 /= norm
		W42 /= norm
		W43 /= norm
		W44 /= norm

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

	#..................FAFT 64  128 ..........................................watch out the axes

	def Fourier_X_To_Lambda_64_128_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dx, self.delta_x,  self.FAFT_segment_axes1, self.FAFT_axes1, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dy, self.delta_y,  self.FAFT_segment_axes3, self.FAFT_axes2, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_Lambda_To_X_64_128_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dx, -self.delta_x,  self.FAFT_segment_axes1, self.FAFT_axes1, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dy, -self.delta_y,  self.FAFT_segment_axes3, self.FAFT_axes2, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_P_To_Theta_64_128_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dp_x, self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dp_y, self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes3, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_Theta_To_P_64_128_GPU(self, W_gpu ):
		cuda_faft128( int(W_gpu.gpudata),  self.dp_x, -self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		cuda_faft64(  int(W_gpu.gpudata),  self.dp_y, -self.delta_p_y,  self.FAFT_segment_axes3, self.FAFT_axes3, self.NF  )
		W_gpu /= W_gpu.size

	#................FAFT 64 64 .............................................[ py , y , px , x ]

	def Fourier_X_To_Lambda_64_64_GPU(self, W_gpu ):
		faft64( int(W_gpu.gpudata), self.dx, self.delta_x, self.FAFT_segment_axes1, self.FAFT_axes3, self.NF )
		faft64( int(W_gpu.gpudata), self.dy, self.delta_y, self.FAFT_segment_axes1, self.FAFT_axes1, self.NF )
		W_gpu /= W_gpu.size

	def Fourier_Lambda_To_X_64_64_GPU(self, W_gpu ):
		faft64( int(W_gpu.gpudata), self.dx, -self.delta_x, self.FAFT_segment_axes1, self.FAFT_axes3, self.NF )
		faft64( int(W_gpu.gpudata), self.dy, -self.delta_y, self.FAFT_segment_axes1, self.FAFT_axes1, self.NF )
		W_gpu /= np.sqrt(W_gpu.size)/100.

	#

	def Fourier_P_To_Theta_64_64_GPU(self, W_gpu ):
		faft64(  int(W_gpu.gpudata),  self.dp_x, self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes2, self.NF  )
		faft64(  int(W_gpu.gpudata),  self.dp_y, self.delta_p_y,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		W_gpu /= W_gpu.size

	def Fourier_Theta_To_P_64_64_GPU(self, W_gpu ):
		faft64(  int(W_gpu.gpudata),  self.dp_x, -self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes2, self.NF  )
		faft64(  int(W_gpu.gpudata),  self.dp_y, -self.delta_p_y,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		W_gpu /= np.sqrt(W_gpu.size)/100.

	def _Fourier_P_To_Theta_64_64_GPU(self, W_gpu ):
		faft64(  int(W_gpu.gpudata),  self.dp_x, self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes2, self.NF  )
		faft64(  int(W_gpu.gpudata),  self.dp_y, self.delta_p_y,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		W_gpu /= W_gpu.size

	def _Fourier_Theta_To_P_64_64_GPU(self, W_gpu ):
		faft64(  int(W_gpu.gpudata),  self.dp_x, -self.delta_p_x,  self.FAFT_segment_axes1, self.FAFT_axes2, self.NF  )
		faft64(  int(W_gpu.gpudata),  self.dp_y, -self.delta_p_y,  self.FAFT_segment_axes1, self.FAFT_axes0, self.NF  )
		W_gpu /= np.sqrt(W_gpu.size)/100.
	
	#...........................................................................................
	
	def Fourier_X_To_Lambda_64_64_4x4_GPU(self, W11, W12, W13, W14,  W21, W22, W23, W24, 
						    W31, W32, W33, W34,  W41, W42, W43, W44 ):
		self.Fourier_X_To_Lambda_64_64_GPU( W11 )
		self.Fourier_X_To_Lambda_64_64_GPU( W12 )
		self.Fourier_X_To_Lambda_64_64_GPU( W13 )
		self.Fourier_X_To_Lambda_64_64_GPU( W14 )

		self.Fourier_X_To_Lambda_64_64_GPU( W21 )
		self.Fourier_X_To_Lambda_64_64_GPU( W22 )
		self.Fourier_X_To_Lambda_64_64_GPU( W23 )
		self.Fourier_X_To_Lambda_64_64_GPU( W24 )

		self.Fourier_X_To_Lambda_64_64_GPU( W31 )
		self.Fourier_X_To_Lambda_64_64_GPU( W32 )
		self.Fourier_X_To_Lambda_64_64_GPU( W33 )
		self.Fourier_X_To_Lambda_64_64_GPU( W34 )

		self.Fourier_X_To_Lambda_64_64_GPU( W41 )
		self.Fourier_X_To_Lambda_64_64_GPU( W42 )
		self.Fourier_X_To_Lambda_64_64_GPU( W43 )
		self.Fourier_X_To_Lambda_64_64_GPU( W44 )

		#print ' Fourier X to Lambda'

	def Fourier_Lambda_To_X_64_64_4x4_GPU(self, W11, W12, W13, W14,  W21, W22, W23, W24, 
						   W31, W32, W33, W34,  W41, W42, W43, W44 ):
		self.Fourier_Lambda_To_X_64_64_GPU( W11 )
		self.Fourier_Lambda_To_X_64_64_GPU( W12 )
		self.Fourier_Lambda_To_X_64_64_GPU( W13 )
		self.Fourier_Lambda_To_X_64_64_GPU( W14 )

		self.Fourier_Lambda_To_X_64_64_GPU( W21 )
		self.Fourier_Lambda_To_X_64_64_GPU( W22 )
		self.Fourier_Lambda_To_X_64_64_GPU( W23 )
		self.Fourier_Lambda_To_X_64_64_GPU( W24 )

		self.Fourier_Lambda_To_X_64_64_GPU( W31 )
		self.Fourier_Lambda_To_X_64_64_GPU( W32 )
		self.Fourier_Lambda_To_X_64_64_GPU( W33 )
		self.Fourier_Lambda_To_X_64_64_GPU( W34 )

		self.Fourier_Lambda_To_X_64_64_GPU( W41 )
		self.Fourier_Lambda_To_X_64_64_GPU( W42 )
		self.Fourier_Lambda_To_X_64_64_GPU( W43 )
		self.Fourier_Lambda_To_X_64_64_GPU( W44 )

	def Fourier_Theta_To_P_64_64_4x4_GPU(self, W11, W12, W13, W14,  W21, W22, W23, W24, 
						   W31, W32, W33, W34,  W41, W42, W43, W44 ):
		self.Fourier_Theta_To_P_64_64_GPU( W11 )
		self.Fourier_Theta_To_P_64_64_GPU( W12 )
		self.Fourier_Theta_To_P_64_64_GPU( W13 )
		self.Fourier_Theta_To_P_64_64_GPU( W14 )

		self.Fourier_Theta_To_P_64_64_GPU( W21 )
		self.Fourier_Theta_To_P_64_64_GPU( W22 )
		self.Fourier_Theta_To_P_64_64_GPU( W23 )
		self.Fourier_Theta_To_P_64_64_GPU( W24 )

		self.Fourier_Theta_To_P_64_64_GPU( W31 )
		self.Fourier_Theta_To_P_64_64_GPU( W32 )
		self.Fourier_Theta_To_P_64_64_GPU( W33 )
		self.Fourier_Theta_To_P_64_64_GPU( W34 )

		self.Fourier_Theta_To_P_64_64_GPU( W41 )
		self.Fourier_Theta_To_P_64_64_GPU( W42 )
		self.Fourier_Theta_To_P_64_64_GPU( W43 )
		self.Fourier_Theta_To_P_64_64_GPU( W44 )	

	def Fourier_P_To_Theta_64_64_4x4_GPU(self, W11, W12, W13, W14,  W21, W22, W23, W24, 
						   W31, W32, W33, W34,  W41, W42, W43, W44 ):
		self.Fourier_P_To_Theta_64_64_GPU( W11 )
		self.Fourier_P_To_Theta_64_64_GPU( W12 )
		self.Fourier_P_To_Theta_64_64_GPU( W13 )
		self.Fourier_P_To_Theta_64_64_GPU( W14 )

		self.Fourier_P_To_Theta_64_64_GPU( W21 )
		self.Fourier_P_To_Theta_64_64_GPU( W22 )
		self.Fourier_P_To_Theta_64_64_GPU( W23 )
		self.Fourier_P_To_Theta_64_64_GPU( W24 )

		self.Fourier_P_To_Theta_64_64_GPU( W31 )
		self.Fourier_P_To_Theta_64_64_GPU( W32 )
		self.Fourier_P_To_Theta_64_64_GPU( W33 )
		self.Fourier_P_To_Theta_64_64_GPU( W34 )

		self.Fourier_P_To_Theta_64_64_GPU( W41 )
		self.Fourier_P_To_Theta_64_64_GPU( W42 )
		self.Fourier_P_To_Theta_64_64_GPU( W43 )
		self.Fourier_P_To_Theta_64_64_GPU( W44 )

	#.........................................................................................

	def _Fourier_Theta_To_P_64_64_4x4_GPU(self, W11, W12, W13, W14,  W21, W22, W23, W24, 
						    W31, W32, W33, W34,  W41, W42, W43, W44 ):
		self._Fourier_Theta_To_P_64_64_GPU( W11 )
		self._Fourier_Theta_To_P_64_64_GPU( W12 )
		self._Fourier_Theta_To_P_64_64_GPU( W13 )
		self._Fourier_Theta_To_P_64_64_GPU( W14 )

		self._Fourier_Theta_To_P_64_64_GPU( W21 )
		self._Fourier_Theta_To_P_64_64_GPU( W22 )
		self._Fourier_Theta_To_P_64_64_GPU( W23 )
		self._Fourier_Theta_To_P_64_64_GPU( W24 )

		self._Fourier_Theta_To_P_64_64_GPU( W31 )
		self._Fourier_Theta_To_P_64_64_GPU( W32 )
		self._Fourier_Theta_To_P_64_64_GPU( W33 )
		self._Fourier_Theta_To_P_64_64_GPU( W34 )

		self._Fourier_Theta_To_P_64_64_GPU( W41 )
		self._Fourier_Theta_To_P_64_64_GPU( W42 )
		self._Fourier_Theta_To_P_64_64_GPU( W43 )
		self._Fourier_Theta_To_P_64_64_GPU( W44 )	

	def _Fourier_P_To_Theta_64_64_4x4_GPU(self, W11, W12, W13, W14,  W21, W22, W23, W24, 
						    W31, W32, W33, W34,  W41, W42, W43, W44 ):
		self._Fourier_P_To_Theta_64_64_GPU( W11 )
		self._Fourier_P_To_Theta_64_64_GPU( W12 )
		self._Fourier_P_To_Theta_64_64_GPU( W13 )
		self._Fourier_P_To_Theta_64_64_GPU( W14 )

		self._Fourier_P_To_Theta_64_64_GPU( W21 )
		self._Fourier_P_To_Theta_64_64_GPU( W22 )
		self._Fourier_P_To_Theta_64_64_GPU( W23 )
		self._Fourier_P_To_Theta_64_64_GPU( W24 )

		self._Fourier_P_To_Theta_64_64_GPU( W31 )
		self._Fourier_P_To_Theta_64_64_GPU( W32 )
		self._Fourier_P_To_Theta_64_64_GPU( W33 )
		self._Fourier_P_To_Theta_64_64_GPU( W34 )

		self._Fourier_P_To_Theta_64_64_GPU( W41 )
		self._Fourier_P_To_Theta_64_64_GPU( W42 )
		self._Fourier_P_To_Theta_64_64_GPU( W43 )
		self._Fourier_P_To_Theta_64_64_GPU( W44 )

	#.........................................................................................
		
	def Fourier_CPU(self, W,a):
		return np.fft.fftshift( np.fft.fft( np.fft.fftshift(W,axes=a) ,axis=a) ,axes=a)

	def iFourier_CPU(self, W,a):
		return np.fft.fftshift( np.fft.ifft( np.fft.fftshift(W,axes=a) ,axis=a) ,axes=a)

	def Fourier2D_CPU(self, W):
		return np.fft.fftshift( np.fft.fft2( np.fft.fftshift(W) ) )

	def iFourier2D_CPU(self, W):
		return np.fft.fftshift( np.fft.ifft2( np.fft.fftshift(W) ) )	


	#

	def FFT_P_To_Theta_CPU(self, W):
		return Fourier_CPU(Fourier_CPU(self, W, 0 ),2)

	def FFT_P_To_Theta_CPU(self, W):
		return iFourier_CPU(iFourier_CPU(self, W, 0 ),2)

	def FFT_X_To_Lambda_CPU(self, W):
		return Fourier_CPU(Fourier_CPU(self, W, 1 ),3)

	def FFT_Lambda_To_X_CPU(self, W):
		return iFourier_CPU(iFourier_CPU(self, W, 1 ),3)


	def FFT_P_To_Theta__4x4_CPU(self, W11, W12, W13, W14,  W21, W22, W23, W24, 
					  W31, W32, W33, W34,  W41, W42, W43, W44):
	
		W11[:] = self.FFT_P_To_Theta_CPU(W11)
		W12[:] = self.FFT_P_To_Theta_CPU(W12)
		W13[:] = self.FFT_P_To_Theta_CPU(W13)
		W14[:] = self.FFT_P_To_Theta_CPU(W14)

		W21[:] = self.FFT_P_To_Theta_CPU(W21)
		W22[:] = self.FFT_P_To_Theta_CPU(W22)
		W23[:] = self.FFT_P_To_Theta_CPU(W23)
		W24[:] = self.FFT_P_To_Theta_CPU(W24)
	
		W31[:] = self.FFT_P_To_Theta_CPU(W31)
		W32[:] = self.FFT_P_To_Theta_CPU(W32)
		W33[:] = self.FFT_P_To_Theta_CPU(W33)
		W34[:] = self.FFT_P_To_Theta_CPU(W34)

		W41[:] = self.FFT_P_To_Theta_CPU(W41)
		W42[:] = self.FFT_P_To_Theta_CPU(W42)
		W43[:] = self.FFT_P_To_Theta_CPU(W43)
		W44[:] = self.FFT_P_To_Theta_CPU(W44)

	def FFT_P_To_Theta__4x4_CPU(self, W11, W12, W13, W14,  W21, W22, W23, W24, 
					  W31, W32, W33, W34,  W41, W42, W43, W44):
	
		W11[:] = self.FFT_Theta_To_P_CPU(W11)
		W12[:] = self.FFT_Theta_To_P_CPU(W12)
		W13[:] = self.FFT_Theta_To_P_CPU(W13)
		W14[:] = self.FFT_Theta_To_P_CPU(W14)

		W21[:] = self.FFT_Theta_To_P_CPU(W21)
		W22[:] = self.FFT_Theta_To_P_CPU(W22)
		W23[:] = self.FFT_Theta_To_P_CPU(W23)
		W24[:] = self.FFT_Theta_To_P_CPU(W24)
	
		W31[:] = self.FFT_Theta_To_P_CPU(W31)
		W32[:] = self.FFT_Theta_To_P_CPU(W32)
		W33[:] = self.FFT_Theta_To_P_CPU(W33)
		W34[:] = self.FFT_Theta_To_P_CPU(W34)

		W41[:] = self.FFT_Theta_To_P_CPU(W41)
		W42[:] = self.FFT_Theta_To_P_CPU(W42)
		W43[:] = self.FFT_Theta_To_P_CPU(W43)
		W44[:] = self.FFT_Theta_To_P_CPU(W44)

		

	#.......................................................................
	
	def Run(self):

		if self.gridDIM_x==128 and self.gridDIM_y == 64:
			Fourier_X_To_Lambda = self.Fourier_X_To_Lambda_64_128_GPU
			Fourier_Lambda_To_X = self.Fourier_Lambda_To_X_64_128_GPU
			Fourier_P_To_Theta  = self.Fourier_P_To_Theta_64_128_GPU
			Fourier_Theta_To_P  = self.Fourier_Theta_To_P_64_128_GPU

		elif self.gridDIM_x==64 and self.gridDIM_y == 64:
			Fourier_X_To_Lambda = self.Fourier_X_To_Lambda_64_64_4x4_GPU
			Fourier_Lambda_To_X = self.Fourier_Lambda_To_X_64_64_4x4_GPU
			Fourier_P_To_Theta  = self.Fourier_P_To_Theta_64_64_4x4_GPU
			Fourier_Theta_To_P  = self.Fourier_Theta_To_P_64_64_4x4_GPU


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

		W11 = self.W11 
		W12 = self.W12 
		W13 = self.W13 
		W14 = self.W14 

		W21 = self.W22 
		W22 = self.W22 
		W23 = self.W23 
		W24 = self.W24 

		W31 = self.W31 
		W32 = self.W32 
		W33 = self.W33 
		W34 = self.W34 
	
		W41 = self.W41 
		W42 = self.W42 
		W43 = self.W43 
		W44 = self.W44 

		average_x   = []
		average_p_x = []

		average_x_square   = []
		average_p_x_square = []
		average_y_square   = []
		average_p_y_square = []

		average_y   = []
		average_p_y = []

		average_Alpha_1 = []
		average_Alpha_2 = []

		average_Alpha_1_x = []
		average_Alpha_2_y = []		

		energyKinetic     = []

		energyPotential   = []
		energy_Mass = []
		energy_p_x  = []
		energy_p_y  = []

		average_D_1_A_0 = []
		average_D_1_A_1 = []
		average_D_1_A_2 = []

		skipFrameTime = []

		self.blockCUDA = (self.gridDIM_x,1,1) 
		self.gridCUDA  = (self.gridDIM_x,self.gridDIM_y,self.gridDIM_y) 


		for tIndex in timeRangeIndex:


			print ' t index = ', tIndex
			t = np.float64(tIndex*self.dt)

			self.boundary_xp_GPU    (W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44) 

			self.Real_GPU(W11)  
			self.Real_GPU(W22)  
			self.Real_GPU(W33)  
			self.Real_GPU(W44)

			self.Normalize_4x4_GPU (W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44)

			if tIndex%self.skipFrame == 0 :
	    
				skipFrameTime.append( tIndex*self.dt )	
            			    
				average_x.append(          self.Average_x_GPU          (W11, W22, W33, W44).get() )
				average_p_x.append(        self.Average_p_x_GPU        (W11, W22, W33, W44).get() )
				average_p_x_square.append( self.Average_p_x_square_GPU (W11, W22, W33, W44).get() )			
				average_x_square.append(   self.Average_x_square_GPU   (W11, W22, W33, W44).get() )

				average_y.append(   	   self.Average_y_GPU  		(W11, W22, W33, W44).get() )
				average_p_y.append( 	   self.Average_p_y_GPU		(W11, W22, W33, W44).get() )		
				average_y_square.append(   self.Average_y_square_GPU    (W11, W22, W33, W44).get() )
				average_p_y_square.append( self.Average_p_y_square_GPU  (W11, W22, W33, W44).get() )

				average_Alpha_1.append( self.Average_Alpha_1_GPU(W14,W41,W23,W32).get()  )	
				average_Alpha_2.append( self.Average_Alpha_2_GPU(W14,W41,W23,W32).get()  )

				average_Alpha_1_x.append( self.Average_Alpha_1_x_GPU(W14,W41,W23,W32).get()  )	
				average_Alpha_2_y.append( self.Average_Alpha_2_y_GPU(W14,W41,W23,W32).get()  )

				energyKinetic.append(
				self.Average_EnergyKinetic_GPU(
				W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44).get())	

				energyPotential.append(
				self.Average_EnergyPotential_GPU(
				W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44,t).get())
			

				average_D_1_A_0.append( 
				self.Average_D_1_A_0_GPU(
				W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44,t).get()) 

				average_D_1_A_1.append(
				self.Average_D_1_A_1_GPU(W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44,t).get()  ) 

				average_D_1_A_2.append(
				self.Average_D_1_A_2_GPU(W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44,t).get()  )


			#p x  ->  p lambda
			Fourier_X_To_Lambda(W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44) 

			#........................Filter.....................
			if self.FilterParticle == 1:
				sign = np.int32(1)
				#print ' pre projection ',self.Norm_4x4_GPU(W11,W22,W33,W44).real
				self.ParticleProjector_GPU( W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44, sign )
				print ' post projection ', self.Norm_4x4_GPU(W11,W22,W33,W44)
			#...................................................

			self.exp_p_lambda_GPU(W11,W12,W13,W14, W21,W22,W23,W24, W31,W32,W33,W34, W41,W42,W43,W44) 
			
			# p lambda  ->  theta x
			Fourier_Lambda_To_X (W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44) 
			Fourier_P_To_Theta  (W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44)  

			self.Potential_Propagator(
				W11,W12,W13,W14, W21,W22,W23,W24, W31,W32,W33,W34, W41,W42,W43,W44,
				block=self.blockCUDA, grid=self.gridCUDA)
			 
			# theta x -> p x
			Fourier_Theta_To_P( W11,W12,W13,W14, W21,W22,W23,W24, W31,W32,W33,W34, W41,W42,W43,W44 )
			
			

			
		#self.Normalize_4x4_GPU( W11,W12,W13,W14,W21,W22,W23,W24,W31,W32,W33,W34,W41,W42,W43,W44 ) 

		self.skipFrameTime = np.array(skipFrameTime)

		self.average_x   = np.array(average_x  )
		self.average_p_x = np.array(average_p_x)
		self.average_y   = np.array(average_y  )
		self.average_p_y = np.array(average_p_y)

		self.average_x_square   = np.array(average_x_square  )
		self.average_p_x_square = np.array(average_p_x_square)
		self.average_y_square   = np.array(average_y_square  )
		self.average_p_y_square = np.array(average_p_y_square)

		self.average_Alpha_1 = np.array(average_Alpha_1).flatten()
		self.average_Alpha_2 = np.array(average_Alpha_2).flatten()

		self.average_Alpha_1_x = np.array(average_Alpha_1_x)
		self.average_Alpha_2_y = np.array(average_Alpha_2_y)

		self.energyKinetic      = np.array(energyKinetic)
		self.energyPotential    = np.array(energyPotential)

		self.energy_Mass = np.array(energy_Mass)
		self.energy_p_x  = np.array(energy_p_x)
		self.energy_p_y  = np.array(energy_p_y)

		self.average_D_1_A_0 = np.array(average_D_1_A_0)
		self.average_D_1_A_1 = np.array(average_D_1_A_1)
		self.average_D_1_A_2 = np.array(average_D_1_A_2)

		self.timeRange = np.array(timeRangeIndex)*self.dt

		self.file['/Ehrenfest/energyKinetic']       = self.energyKinetic
		self.file['/Ehrenfest/energyPotential']     = self.energyPotential
		self.file['/Ehrenfest/average_x']    = self.average_x
		self.file['/Ehrenfest/average_p_x']  = self.average_p_x
		self.file['/Ehrenfest/average_y']    = self.average_y
		self.file['/Ehrenfest/average_p_y']  = self.average_p_y

		self.file['/Ehrenfest/average_x_square'  ]  = self.average_x_square
		self.file['/Ehrenfest/average_p_x_square']  = self.average_p_x_square
		self.file['/Ehrenfest/average_y_square'  ]  = self.average_y_square
		self.file['/Ehrenfest/average_p_y_square']  = self.average_p_y_square

			

