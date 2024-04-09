#include "FerroE.H"

//Compute x-component of electric field corresponding to the Landau free energy contribution
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real FerroE::compute_ex_Landau(const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz)
{
    return alpha_1*Px + alpha_11*std::pow(Px,3.) + alpha_111*std::pow(Px,5.)
               + 2. * alpha_12 * Px * std::pow(Py,2.)
               + 2. * alpha_12 * Px * std::pow(Pz,2.)
               + 4. * alpha_112 * std::pow(Px,3.) * (std::pow(Py,2.) + std::pow(Pz,2.))
               + 2. * alpha_112 * Px * std::pow(Py,4.)
               + 2. * alpha_112 * Px * std::pow(Pz,4.)
               + 2. * alpha_123 * Px * std::pow(Py,2.) * std::pow(Pz,2.);
}


//Compute y-component of electric field corresponding to the Landau free energy contribution
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real FerroE::compute_ey_Landau(const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz)
{
    return alpha_1*Py + alpha_11*std::pow(Py,3.) + alpha_111*std::pow(Py,5.)
               + 2. * alpha_12 * Py * std::pow(Px,2.)
               + 2. * alpha_12 * Py * std::pow(Pz,2.)
               + 4. * alpha_112 * std::pow(Py,3.) * (std::pow(Px,2.) + std::pow(Pz,2.))
               + 2. * alpha_112 * Py * std::pow(Px,4.)
               + 2. * alpha_112 * Py * std::pow(Pz,4.)
               + 2. * alpha_123 * Py * std::pow(Px,2.) * std::pow(Pz,2.);
}


//Compute z-component of electric field corresponding to the Landau free energy contribution
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real FerroE::compute_ez_Landau(const amrex::Real Px, const amrex::Real Py, const amrex::Real Pz)
{
    return alpha_1*Pz + alpha_11*std::pow(Pz,3.) + alpha_111*std::pow(Pz,5.)
               + 2. * alpha_12 * Pz * std::pow(Px,2.)
               + 2. * alpha_12 * Pz * std::pow(Py,2.)
               + 4. * alpha_112 * std::pow(Pz,3.) * (std::pow(Px,2.) + std::pow(Py,2.))
               + 2. * alpha_112 * Pz * std::pow(Px,4.)
               + 2. * alpha_112 * Pz * std::pow(Py,4.)
               + 2. * alpha_123 * Pz * std::pow(Px,2.) * std::pow(Py,2.);
}

